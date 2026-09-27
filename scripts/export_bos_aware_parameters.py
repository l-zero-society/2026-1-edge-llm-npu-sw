#!/usr/bin/env python3
"""Export frozen measured policies through the existing, unchanged QB exporter."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from static_quant.core import Options,quantize_input,exact_accumulator
from static_quant.cli import operation_options
from static_quant.export import Exporter,reload_parameters
from static_quant.hardware import Profile
from static_quant.gemma import input_group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='diagnostics/bos_aware')
    p.add_argument('--output-dir',default='diagnostics/bos_aware/calibrated')
    p.add_argument('--policy',default='stratified_64')
    p.add_argument('--cache-dir',default='/tmp/bos-aware-capture')
    p.add_argument('--allow-unverified-export',action='store_true')
    args=p.parse_args();source=Path(args.results);art=ROOT/'calibration_outputs/gguf-all-126-oasst1'
    old=json.loads((art/'manifest.json').read_text());identity=json.loads((source/'provenance.json').read_text())
    entries={e['module_name']:e for e in old['modules']};profile=Profile()
    metadata=dict(model=old['gguf']['file_name'],source_format='reconstructed Q8_0 GGUF',
                  datasets=old['datasets'],source_commit=identity['commit'],source_tree_dirty=True,
                  calibration_policy=args.policy,diagnostic_provenance=identity,
                  gguf_source_error='not measured',selection_split='calibration')
    exporter=Exporter(args.output_dir,profile,metadata,allow_unverified=args.allow_unverified_export)
    for file in sorted((source/'results').glob('*.json')):
        data=json.loads(file.read_text());name=file.stem
        if args.policy not in data['searches']:continue
        search=data['searches'][args.policy];choice=search['selected'];entry=entries[name]
        directory=art/Path(entry['module_manifest']).parent
        with np.load(directory/'scales.npz') as arrays:sw=arrays['op0000.s_W'].copy()
        params=profile.approximate(choice['s_X']*sw/choice['s10'])
        options=operation_options(name,Options(mode='mse',search_rows=search['sampling']['capacity'],seed=42))
        operation=SimpleNamespace(name=name,n=data['N'],k=data['K'],tokens=old['datasets']['calibration']['valid_tokens'],
            sx=choice['s_X'],s10=choice['s10'],sw=sw,params=params,options=options)
        x=np.load(Path(args.cache_dir)/str(data['layer'])/'calibration'/(input_group(name)+'.npy'),mmap_mode='r')[:4]
        wq=np.load(art/entry['int8_weight_file'],mmap_mode='r')[:16]
        acc=exact_accumulator(quantize_input(x,operation.sx),wq)
        q=profile.apply(acc,params['multiplier'][:16],params['shift'][:16]);words=profile.pack(params['multiplier'],params['shift'])
        vectors=[dict(channel=c,accumulator=int(acc[r,c]),word=int(words[c]),expected_int10=int(q[r,c]),lut_index=int(q[r,c])&1023)
                 for r in range(len(x)) for c in range(len(wq))]
        report=dict(selection=search,s_X=operation.sx,s_10=operation.s10)
        for split in ['calibration','validation']:
            measured=next(r for r in data['evaluation'] if r['policy']==args.policy and r['split']==split)
            report[split]=dict(measured=measured,local_total={'mse':measured['all_final_local_total_mse'],'nmse':measured['all_final_local_total_nmse']})
        exporter.add(operation,report,vectors,input_group=input_group(name))
    manifest=exporter.finish()
    for entry in manifest['modules']:
        p,params,_=reload_parameters(args.output_dir,entry['module_name'])
        assert entry['s_10']>0 and np.all(params[2]==0)
    print('Exported and reloaded',len(manifest['modules']),'static operation policies;',manifest['binary_export']['status'])


if __name__=='__main__':main()
