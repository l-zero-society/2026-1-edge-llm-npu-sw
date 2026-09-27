#!/usr/bin/env python3
"""Verify complete outputs, frozen policies, fit/evaluation separation and replay."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'diagnostics'


def main():
    def read(name):return list(csv.DictReader((D/name).open()))
    def close(a,b):
        if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):raise AssertionError((a,b))
    main_rows=read('quant_error_attribution.csv');stats=read('down_proj_activation_stats.csv')
    mlp=read('mlp_activation_comparison.csv');inputs=read('down_proj_scale_sweep.csv');outputs=read('down_proj_output_scale_sweep.csv')
    assert [len(x) for x in [main_rows,stats,mlp,inputs,outputs]]==[252,36,108,252,216]
    indexed={(r['module_name'],r['split']):r for r in main_rows};assert len(indexed)==252
    assert len({(r['layer'],r['split']) for r in stats})==36
    assert len({(r['layer'],r['module'],r['split']) for r in mlp})==108
    assert len({(r['layer'],r['policy'],r['split']) for r in inputs})==252
    assert len({(r['layer'],r['s_10'],r['split']) for r in outputs})==216
    parameters=read('quantization_parameters.csv')
    assert len(parameters)==126 and len({r['module_name'] for r in parameters})==126
    for r in parameters:
        base=indexed[r['module_name'],'calibration']
        close(r['s_X'],base['s_X']);close(r['s_10'],base['s_10'])
        assert int(r['multiplier_min'])>=0 and int(r['multiplier_max'])<=65535
        assert int(r['shift_min'])>=0 and int(r['shift_max'])<=31
    manifest=json.loads((ROOT/'calibration_outputs/gguf-all-126-oasst1/manifest.json').read_text())
    for e in manifest['modules']:
        for split,tokens in [('calibration',34221),('validation',10141)]:
            r=indexed[e['module_name'],split]
            close(r['s_X'],e['s_X']);close(r['s_10'],e['s_10']);assert int(r['valid_tokens'])==tokens
            assert float(r['stored_mse_relative_difference'])<2e-5
            for prefix in ['input_only','weight_only','int8_pair','ideal_int10','actual_requant','final_local_total','ideal_requantization_only','requantization_only']:
                close(r[prefix+'_nmse'],float(r[prefix+'_mse'])/float(r[prefix+'_reference_energy']))
                assert float(r[prefix+'_mse'])>=0 and float(r[prefix+'_mae'])>=0
            assert float(r['actual_requant_max_absolute_error'])<=float(r['s_10'])*(1+1e-10)
    for r in inputs:
        counterpart=next(c for c in inputs if c['layer']==r['layer'] and c['policy']==r['policy'] and c['split']=='calibration')
        close(r['threshold'],counterpart['threshold']);close(r['s_X'],counterpart['s_X'])
        base=indexed[r['module_name'],r['split']];close(r['s_10'],base['s_10'])
        assert int(r['bad_ratio_channels'])==0
        if r['policy']=='absmax':
            for k in ['input_only_mse','int8_pair_mse','final_local_total_mse']:close(r[k],base[k])
    for r in outputs:
        base=indexed[r['module_name'],r['split']];close(r['s_X'],base['s_X'])
        if r['selected_current']=='True':
            for k in ['final_local_total_mse','requantization_only_mse','int10_clip_rate']:close(r[k],base[k])
    for r in stats:
        base=indexed[r['module_name'],r['split']];close(r['zero_rate'],base['activation_zero_rate'])
        assert float(r['below_half_scale_rate'])<=float(r['zero_rate'])
        assert float(r['clipping_rate'])<=float(r['saturation_rate'])
    identity=json.loads((D/'diagnosis_provenance.json').read_text())['identity']
    for file,digest in identity['source_hashes'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    audit=json.loads((D/'outlier_position_audit.json').read_text())
    assert audit['token_ids_verified'] and audit['model_sha256']==identity['model_sha256']
    for layer in audit['layers']:
        assert layer['splits']['calibration']['historical_search_scores_reproduced']
        assert layer['splits']['calibration']['sampled_bos_rows']==0
        assert all(s['all_int10_clips_at_bos'] for s in layer['splits'].values())
    result=dict(passed=True,operations=126,attribution_rows=len(main_rows),down_stats_rows=len(stats),
                mlp_rows=len(mlp),input_sweep_rows=len(inputs),output_sweep_rows=len(outputs),
                max_stored_mse_relative_difference=max(float(r['stored_mse_relative_difference']) for r in main_rows),
                scope='Output coverage/consistency, frozen baseline scales, calibration-only threshold fitting, source hash identity; not RTL or model-accuracy validation')
    (D/'output_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
