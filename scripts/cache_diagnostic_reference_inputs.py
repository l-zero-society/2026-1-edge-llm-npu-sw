#!/usr/bin/env python3
"""Optional offline bulk capture: one ORIGINAL-model prefix per sample.

Writes to a separate staging directory and publishes only future-layer caches.
Does not compute metrics or modify any model/quantization parameter. This is an
optional execution optimization; diagnose_linear_quant_error.py is standalone.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from static_quant.gemma import input_group,sha256_file
from quant_diagnostic_source import RecordedGGUFSource
from diagnose_linear_quant_error import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--artifacts',default='calibration_outputs/gguf-all-126-oasst1')
    p.add_argument('--data-dir',default='calibration_outputs/data-oasst1-v1')
    p.add_argument('--output-dir',default='diagnostics')
    p.add_argument('--cache-dir',default='/tmp/quant-diagnosis-capture')
    p.add_argument('--staging-dir',default='/tmp/quant-diagnosis-bulk-capture')
    p.add_argument('--first-layer',type=int,default=6)
    args=p.parse_args()
    if not 0<=args.first_layer<18:raise ValueError('invalid first layer')
    manifest=json.loads((Path(args.artifacts)/'manifest.json').read_text())
    source=RecordedGGUFSource(args.model,args.data_dir,manifest,threads=2)
    groups={}
    for entry in manifest['modules']:
        name=entry['module_name'];layer=int(name.split('.')[2])
        if layer>=args.first_layer:groups.setdefault(input_group(name),(name,layer))
    provenance=dict(argv=sys.argv,script_sha256=sha256_file(__file__),source_sha256=sha256_file(ROOT/'scripts/quant_diagnostic_source.py'),
                    model_sha256=sha256_file(args.model),started=time.time(),scope='original model inputs; hooks do not replace outputs')
    for split in ['calibration','validation']:
        arrays={};offsets={g:0 for g in groups};tokens=sum(map(len,source.datasets[split]))
        for g,(name,layer) in groups.items():
            directory=Path(args.staging_dir)/str(layer)/split;directory.mkdir(parents=True,exist_ok=True)
            arrays[g]=np.lib.format.open_memmap(directory/(g+'.npy'),mode='w+',dtype=np.float32,shape=(tokens,source.modules[name].weight.shape[1]))
        class StopCapture(Exception):pass
        handles=[]
        def make_hook(g,name):
            def hook(module,inputs):
                x=inputs[0].detach().numpy().reshape(-1,arrays[g].shape[1])
                if not np.isfinite(x).all():raise ValueError('nonfinite reference input')
                pos=offsets[g];arrays[g][pos:pos+len(x)]=x;offsets[g]+=len(x)
                if name=='model.layers.17.mlp.down_proj':raise StopCapture()
            return hook
        for g,(name,layer) in groups.items():handles.append(source.modules[name].register_forward_pre_hook(make_hook(g,name)))
        try:
            with source.torch.inference_mode():
                for index,ids in enumerate(source.datasets[split]):
                    x=source.torch.tensor([ids],dtype=source.torch.int64)
                    try:source.model(input_ids=x,attention_mask=source.torch.ones_like(x),use_cache=False)
                    except StopCapture:pass
                    else:raise RuntimeError('stop hook not reached')
                    if (index+1)%16==0:
                        for a in arrays.values():a.flush()
                        print(split,index+1,'samples',flush=True)
        finally:
            for h in handles:h.remove()
        if any(v!=tokens for v in offsets.values()):raise ValueError('capture counts differ')
        for a in arrays.values():a.flush()
        arrays.clear()
        for layer in range(args.first_layer,18):
            completed=[int(p.stem.split('.')[2]) for p in (Path(args.output_dir)/'results').glob('model.layers.*.json')]
            if completed and layer<=max(completed)+1:
                print('Skip publication near active layer',layer,flush=True);continue
            directory=Path(args.cache_dir)/str(layer)/split;directory.mkdir(parents=True,exist_ok=True)
            expected=dict(model=provenance['model_sha256'],data=sha256_file(Path(args.data_dir)/(split+'.jsonl')),
                          source=provenance['source_sha256'],tokens=tokens)
            marker=directory/'identity.json'
            if marker.exists() and json.loads(marker.read_text())!=expected:raise ValueError('cache identity conflict')
            staging=Path(args.staging_dir)/str(layer)/split
            for file in staging.glob('*.npy'):
                target=directory/file.name
                if not target.exists():file.rename(target)
            write_json(directory/'bulk_capture_provenance.json',dict(provenance,split=split,completed=time.time()))
            write_json(marker,expected)
            print('Published',layer,split,flush=True)
    provenance['completed']=time.time()
    write_json(Path(args.output_dir)/'bulk_capture_provenance.json',provenance)


if __name__=='__main__':main()
