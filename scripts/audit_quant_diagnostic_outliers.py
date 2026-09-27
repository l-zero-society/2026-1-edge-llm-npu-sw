#!/usr/bin/env python3
"""Locate special-token outliers and reproduce historical 64-row output search.

Read-only supplementary audit of layers 8 and 17; no new scale policy is fitted.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from static_quant.gemma import sha256_file,load_texts
from static_quant.core import Reservoir,quantize_input
from static_quant.hardware import Profile
from quant_diagnostic_math import configure_blas,integer_dot
from diagnose_linear_quant_error import write_json


def main():
    import csv,gguf
    from transformers import GemmaTokenizerFast
    from transformers.integrations.ggml import GGUFGemmaConverter,GGUF_TOKENIZER_MAPPING,_gguf_parse_value
    model=Path(sys.argv[1]);out=ROOT/'diagnostics';art=ROOT/'calibration_outputs/gguf-all-126-oasst1'
    manifest=json.loads((art/'manifest.json').read_text())
    assert sha256_file(model)==manifest['gguf']['sha256']
    reader=gguf.GGUFReader(str(model));tokenizer_data={}
    for key,target in GGUF_TOKENIZER_MAPPING['tokenizer'].items():
        field=reader.fields.get('tokenizer.'+key)
        if field is not None:
            values=[_gguf_parse_value(field.parts[i],field.types) for i in field.data]
            tokenizer_data[target]=values if field.types[0]==9 else values[0]
    converter=GGUFGemmaConverter(tokenizer_data)
    tokenizer=GemmaTokenizerFast(tokenizer_object=converter.converted(),add_bos_token=True,add_eos_token=False,**converter.additional_kwargs)
    starts={}
    for split in ['calibration','validation']:
        path=ROOT/'calibration_outputs/data-oasst1-v1'/f'{split}.jsonl';meta=manifest['datasets'][split]
        assert sha256_file(path)==meta['file_sha256']
        texts,_=load_texts(path,meta['actual_samples'])
        ids=[tokenizer(t,add_special_tokens=True,truncation=True,max_length=512)['input_ids'] for t in texts]
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==meta['token_ids_sha256']
        assert all(row[0]==2 for row in ids)
        starts[split]=np.concatenate([[0],np.cumsum([len(row) for row in ids[:-1]])]).astype(np.int64)
    configure_blas('accelerate');profile=Profile()
    baseline={(r['layer'],r['split']):r for r in csv.DictReader((out/'quant_error_attribution.csv').open()) if r['module']=='down_proj'}
    result=dict(script_sha256=sha256_file(__file__),argv=sys.argv,model_sha256=manifest['gguf']['sha256'],
                token_ids_verified=True,layers=[])
    for layer in [8,17]:
        name=f'model.layers.{layer}.mlp.down_proj';index=7*layer+6;directory=art/'modules'/f'{index:04d}'
        report=json.loads((directory/'report.json').read_text())['modules'][name]
        scales=np.load(directory/'scales.npz');params=np.load(directory/'qparams.npz')
        sx=float(scales['op0000.s_X']);sw=scales['op0000.s_W'];wq=np.load(directory/'op0000.weights.int8.npy',mmap_mode='r')
        entry=dict(layer=layer,s_X=sx,s_10=float(scales['op0000.s_10']),splits={})
        for split in ['calibration','validation']:
            x=np.load(Path('/tmp/quant-diagnosis-capture')/str(layer)/split/(name+'.input.npy'),mmap_mode='r')
            bos=starts[split];non_bos=np.ones(len(x),dtype=bool);non_bos[bos]=False
            largest_other=0.
            for i in range(0,len(x),128):
                peaks=np.abs(x[i:i+128]).max(axis=1)
                if non_bos[i:i+len(peaks)].any():largest_other=max(largest_other,float(peaks[non_bos[i:i+len(peaks)]].max()))
            bos_x=np.asarray(x[bos],np.float64);bos_absmax=float(np.abs(bos_x).max())
            acc=integer_dot(quantize_input(bos_x,sx),wq,'fp64-exact')
            raw=profile.apply(acc,params['op0000.multiplier'],params['op0000.shift'],saturate=False)
            clipped_bos=int(((raw < -512)|(raw > 511)).sum())
            b=baseline[str(layer),split];clipped_total=round(float(b['int10_clip_rate'])*int(b['output_elements']))
            record=dict(bos_rows=len(bos),bos_absmax=bos_absmax,non_bos_absmax=largest_other,
                        baseline_int10_clip_count=clipped_total,bos_int10_clip_count=clipped_bos,
                        all_int10_clips_at_bos=clipped_bos==clipped_total)
            if split=='calibration':
                reservoir=Reservoir(64,42)
                for i in range(0,len(x),512):reservoir.add(np.arange(i,min(i+512,len(x)))[:,None])
                selected=reservoir.values[:,0].astype(np.int64)
                sample=np.asarray(x[selected],np.float64)
                sa=integer_dot(quantize_input(sample,sx),wq,'fp64-exact');pair=sa*(sx*sw)
                candidates=[]
                for c in report['selection']['candidates']:
                    s=c['s10'];p=profile.approximate(sx*sw/s);hw=profile.apply(sa,p['multiplier'],p['shift'])*s
                    measured=float(np.square(hw-pair).mean())
                    candidates.append(dict(s_10=s,stored_search_mse=c['mse'],reproduced_search_mse=measured,
                        matches=bool(np.isclose(measured,c['mse'],rtol=1e-12,atol=1e-14))))
                record.update(search_rows=selected.tolist(),sampled_bos_rows=int(np.isin(selected,bos).sum()),
                              historical_search_scores_reproduced=all(c['matches'] for c in candidates),search_candidates=candidates)
            entry['splits'][split]=record
            print(layer,split,bos_absmax,largest_other,clipped_bos,clipped_total,flush=True)
        result['layers'].append(entry)
    write_json(out/'outlier_position_audit.json',result)


if __name__=='__main__':main()
