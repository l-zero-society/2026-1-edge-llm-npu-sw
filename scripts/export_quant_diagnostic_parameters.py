#!/usr/bin/env python3
"""Summarize verified stored quantization parameters without changing them."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    artifact=ROOT/'calibration_outputs/gguf-all-126-oasst1'
    manifest=json.loads((artifact/'manifest.json').read_text())
    rows=[]
    for entry in manifest['modules']:
        directory=artifact/Path(entry['module_manifest']).parent
        module_manifest=json.loads((directory/'manifest.json').read_text())
        for name in ['scales.npz','qparams.npz']:
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==module_manifest['artifact_sha256'][name]
        name=entry['module_name']
        report=json.loads((artifact/entry['report_file']).read_text())['modules'][name]
        with np.load(directory/'scales.npz') as scales, np.load(directory/'qparams.npz') as params:
            row=dict(layer=int(name.split('.')[2]),module=name.split('.')[-1],module_name=name,
                     s_X=float(scales['op0000.s_X']),s_10=float(scales['op0000.s_10']),
                     output_scale_policy=report['selection']['mode'],zero_point=0,
                     input_range='[-127,127]',weight_range='[-127,127]',output_range='[-512,511]',
                     accumulator='INT32',rounding='nearest-even',
                     scale_file=str((directory/'scales.npz').relative_to(ROOT)),
                     parameter_file=str((directory/'qparams.npz').relative_to(ROOT)))
            for key,values in [('s_W',scales['op0000.s_W']),('multiplier',params['op0000.multiplier']),
                               ('shift',params['op0000.shift']),('ratio',params['op0000.ratio'])]:
                row[key+'_min']=values.min().item();row[key+'_max']=values.max().item()
            row['output_channels']=len(scales['op0000.s_W'])
            assert np.all(params['op0000.zero_point']==0)
            rows.append(row)
    destination=ROOT/'diagnostics/quantization_parameters.csv'
    with destination.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} verified operation parameter summaries')


if __name__=='__main__':main()
