#!/usr/bin/env python3
"""Discrete cross-layer E2E selection for absmax down_proj calibration."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import absmax_joint_calibration as joint
import run_absmax_joint_stability as stability

pilot=stability.pilot; final=stability.final; base=stability.base; row=stability.row
OUT=ROOT/'diagnostics/absmax_cross_layer_calibration'; WORK=OUT/'work'
PROGRESS=ROOT/'cross_layer_calibration_progress.txt'
SEED=20261003; SELECTION_CHATS=4; FINAL_CHATS=6
SELECTION_TARGETS=24; FINAL_TARGETS=32; MAX_SHORTLIST=6; COORDINATE_PASSES=2
START=time.monotonic(); EPS=1e-15


def progress(message):
    with PROGRESS.open('a') as handle: handle.write(message+'\n');handle.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def deterministic_fresh_split(examples,excluded):
    fresh=[e for e in examples if e['index'] not in excluded and len(e['targets'])>=FINAL_TARGETS]
    fresh.sort(key=lambda e:(hashlib.sha256(f'{SEED}:{e["index"]}'.encode()).digest(),e['index']))
    if len(fresh)!=10: raise RuntimeError(f'expected exactly 10 fresh conversations, got {len(fresh)}')
    selection,held=fresh[:SELECTION_CHATS],fresh[SELECTION_CHATS:]
    if len(selection)!=4 or len(held)!=6 or {e['index'] for e in selection}&{e['index'] for e in held}:
        raise RuntimeError('invalid deterministic fresh split')
    return selection,held


def previous_validation_indices():
    result=set()
    for directory in ('absmax_calibration_pilot','absmax_joint_calibration_pilot','absmax_joint_stability'):
        path=ROOT/'diagnostics'/directory/'validation_per_conversation.csv'
        with path.open(newline='') as handle: result.update(int(r['source_index']) for r in csv.DictReader(handle))
    return result


def config_key(record):
    return (float(record['sx']).hex(),float(record['s10']).hex(),
            tuple(record['multiplier']),tuple(record['shift']))


def deduplicate_candidates(records):
    unique={}
    for record in records:
        key=config_key(record)
        if key in unique:
            unique[key]['sources']=sorted(set(unique[key]['sources']+record['sources']))
        else: unique[key]=dict(record,sources=sorted(set(record['sources'])))
    return list(unique.values())


def classify_sensitivity(old,candidate):
    feasible=candidate['nmse']<=1.05*old['nmse']+EPS and candidate['top1']>=old['top1']-.01-EPS
    if not feasible:return 'UNSAFE'
    return 'BENEFICIAL' if candidate['kl']<old['kl']-EPS else 'NON_BENEFICIAL'


def changed_count(specs,frozen):
    return sum(specs[n]['sx']!=frozen[n]['sx'] or specs[n]['s10']!=frozen[n]['s10'] for n in specs)


def movement(specs,frozen):
    return sum(abs(math.log2(specs[n]['sx']/frozen[n]['sx']))+
               abs(math.log2(specs[n]['s10']/frozen[n]['s10'])) for n in specs)


def candidate_can_replace(incumbent,candidate):
    return (candidate['nmse']<=1.05*incumbent['nmse']+EPS and
            candidate['top1']>=incumbent['top1']-.01-EPS and
            candidate['kl']<incumbent['kl']-EPS)


def choose_coordinate(incumbent_metrics,incumbent_specs,options,frozen):
    feasible=[o for o in options if candidate_can_replace(incumbent_metrics,o['metrics'])]
    if not feasible:return None
    return min(feasible,key=lambda o:(o['metrics']['kl'],o['metrics']['nmse'],-o['metrics']['top1'],
        -o['metrics']['cosine'],changed_count(o['specs'],frozen),movement(o['specs'],frozen),o['candidate_id']))


def checkpoint_valid(path,key):
    return path.exists() and json.loads(path.read_text()).get('checkpoint_key')==key


def frozen_fingerprint(frozen):
    protected=[final.OUT,pilot.OUT,ROOT/'diagnostics/absmax_joint_calibration_pilot',
               ROOT/'diagnostics/absmax_joint_stability']
    return dict(decisions_sha256=base.sha256_file(final.OUT/'decisions.json'),
                parameters_sha256=pilot.compare.fingerprint(frozen),
                artifacts_sha256=pilot.fingerprint_tree(protected))


def params_for(old,sx,s10,multiplier,shift):
    sw=np.asarray(old['params']['ratio'])*old['s10']/old['sx']
    params=joint.Profile().approximate(float(sx)*sw/float(s10))
    np.testing.assert_array_equal(params['multiplier'],multiplier)
    np.testing.assert_array_equal(params['shift'],shift)
    return params


def record_from_candidate(candidate,sources,old):
    return dict(sources=list(sources),sx=float(candidate['sx']),s10=float(candidate['s10']),
        sx_j=int(candidate['sx_j']),s10_i=int(candidate['s10_i']),
        multiplier=candidate['multiplier'].tolist(),shift=candidate['shift'].tolist(),
        local_score=float(candidate['score']),local_relative_change=float((candidate['score']-old['score'])/old['score']) if old['score'] else 0.,
        sx_boundary_hit=bool(candidate['sx_boundary_hit']),s10_boundary_hit=bool(candidate['s10_boundary_hit']),
        effective_shift_min=int(candidate['effective_shift_min']),
        effective_shift_max=int(candidate['effective_shift_max']))


def candidate_to_spec(old,record):
    return dict(old,sx=record['sx'],s10=record['s10'],
                params=params_for(old,record['sx'],record['s10'],record['multiplier'],record['shift']))


def find_candidate(candidates,sx,s10):
    matches=[c for c in candidates if c['feasible'] and c['sx']==sx and c['s10']==s10]
    if len(matches)!=1:raise RuntimeError('stored candidate absent from fixed local grid')
    return matches[0]


def generate_pool(source,frozen,captured,run_key):
    pilot_params=json.loads((ROOT/'diagnostics/absmax_joint_calibration_pilot/proposed_parameters.json').read_text())
    stability_params=json.loads((ROOT/'diagnostics/absmax_joint_stability/proposed_parameters.json').read_text())
    pool={}
    for layer in range(18):
        checkpoint=WORK/f'candidate_layer_{layer:02d}.json'
        key=digest(dict(run=run_key,layer=layer,phase='candidate_pool'))
        if checkpoint_valid(checkpoint,key): records=json.loads(checkpoint.read_text())['candidates']
        else:
            name=f'model.layers.{layer}.mlp.down_proj';old=frozen[name]
            result=joint.calibrate_layer_joint(source.weight(name),old,captured[layer],
                                               sx_js=stability.SX_JS,s10_is=stability.S10_IS)
            candidates=result['candidates'];feasible=[c for c in candidates if c['feasible']]
            old_c=result['old']; non_old=[c for c in feasible if not c['is_old_pair']]
            non_old.sort(key=lambda c:joint.candidate_key(c,old['sx'],old['s10']))
            chosen=[record_from_candidate(old_c,['old'],old_c)]
            if non_old:chosen.append(record_from_candidate(non_old[0],['local_best'],old_c))
            if len(non_old)>1:chosen.append(record_from_candidate(non_old[1],['local_second'],old_c))
            p=pilot_params[name];pc=find_candidate(candidates,float(p['sx_base']),float(p['s10']))
            chosen.append(record_from_candidate(pc,['pilot'],old_c))
            s=stability_params[name];sc=find_candidate(candidates,float(s['sx_base']),float(s['s10']))
            chosen.append(record_from_candidate(sc,['stability'],old_c))
            records=deduplicate_candidates(chosen)
            for index,r in enumerate(records):r['candidate_id']=f'L{layer}C{index}'
            base.write_json(checkpoint,dict(checkpoint_key=key,candidates=records))
        pool[str(layer)]=records;progress(f'CANDIDATE_LAYER {layer}/18 DONE')
    return pool


def capture_rows(frozen,before,pilot_hash):
    stored=json.loads((stability.WORK/'configuration.json').read_text())
    stable_frozen=dict(decisions_sha256=base.sha256_file(final.OUT/'decisions.json'),
        parameters_sha256=pilot.compare.fingerprint(frozen),
        artifacts_sha256=pilot.fingerprint_tree(
            [final.OUT,pilot.OUT,ROOT/'diagnostics/absmax_joint_calibration_pilot']))
    if stored.get('frozen')!=stable_frozen:
        raise RuntimeError('stability capture frozen fingerprint mismatch')
    stable_config=stability.configuration(stable_frozen,pilot_hash)
    if stored!=stable_config:raise RuntimeError('stability capture configuration/fingerprint mismatch')
    capture_meta=stability.WORK/'capture.json';capture_path=stability.WORK/'capture.npz'
    if not stability.checkpoint_matches(capture_meta,stability.config_fingerprint(stable_config)) or not capture_path.exists():
        raise RuntimeError('compatible stability capture checkpoint unavailable')
    return stability.load_capture(capture_path)


def make_split_artifact(selection,held,excluded):
    def item(e):return {k:e[k] for k in ('index','anchor_id','response_id','tree_id') if k in e}
    artifact=dict(seed=SEED,available_fresh_conversations=10,
        previous_validation_source_indices=sorted(excluded),selection=[item(e) for e in selection],
        final=[item(e) for e in held],selection_conversations=4,selection_targets=96,
        final_conversations=6,final_targets=192)
    artifact['fingerprint']=digest(artifact);return artifact


def generate_fp_reference(source,examples,split_fp,run_key):
    records=[]
    progress('SELECTION_FP_START')
    for number,e in enumerate(examples):
        path=WORK/f'selection_fp_{number:02d}.npz';meta=WORK/f'selection_fp_{number:02d}.json'
        key=digest(dict(run=run_key,split=split_fp,phase='selection_fp',source=e['index']))
        if checkpoint_valid(meta,key) and path.exists(): payload=json.loads(meta.read_text())
        else:
            cache=None;logits=[];score=pilot.response.Scores()
            for t,target in enumerate(e['targets'][:SELECTION_TARGETS]):
                ids=e['prompt_ids'] if t==0 else [e['targets'][t-1]]
                value,_,cache=final.cached.run_fp(source,ids,cache,layers=False)
                logits.append(value.numpy().astype(np.float32));score.add(source.torch,value,value,[target])
            np.savez_compressed(path,logits=np.stack(logits),targets=np.asarray(e['targets'][:SELECTION_TARGETS],np.int64))
            payload=dict(checkpoint_key=key,source_index=e['index'],scores=score.raw(),sha256=base.sha256_file(path))
            base.write_json(meta,payload)
        records.append(payload)
    fp_hash=digest([(r['source_index'],r['sha256']) for r in records]);progress('SELECTION_FP_DONE')
    return records,fp_hash


def eval_selection(source,specs,examples,fp_hash,split_fp,phase,layer,candidate_id,incumbent_fp):
    spec_fp=pilot.compare.fingerprint(specs)
    key=digest(dict(split=split_fp,fp_reference=fp_hash,configuration=spec_fp,
                    incumbent=incumbent_fp,phase=phase,layer=layer,candidate=candidate_id))
    checkpoint=WORK/f'eval_{key}.json'
    if checkpoint_valid(checkpoint,key):return json.loads(checkpoint.read_text())['metrics']
    raws=[]
    for number,e in enumerate(examples):
        with np.load(WORK/f'selection_fp_{number:02d}.npz') as saved:
            refs=saved['logits'];targets=saved['targets']
        score=pilot.response.Scores();cache=None
        with patch.object(row,'select_rows',pilot.select_absmax):
            for t,target in enumerate(targets):
                ids=e['prompt_ids'] if t==0 else [int(targets[t-1])]
                q,_,cache=final.cached.run_mode(source,specs,ids,cache,layers=False)
                ref=source.torch.from_numpy(refs[t]);score.add(source.torch,q,ref,[int(target)])
        raws.append(score.raw())
    metrics=pilot.response.merged(raws);base.write_json(checkpoint,dict(checkpoint_key=key,metrics=metrics))
    return metrics


def apply_candidate(specs,frozen,layer,record):
    result=dict(specs);name=f'model.layers.{layer}.mlp.down_proj'
    result[name]=candidate_to_spec(frozen[name],record);return result


def final_conversation(source,modes,e,position):
    scores={m:pilot.response.Scores() for m in modes};caches={};logits={}
    for mode,specs in modes.items():
        if mode=='FP':logits[mode],_,caches[mode]=final.cached.run_fp(source,e['prompt_ids'],None,layers=False)
        else:
            with patch.object(row,'select_rows',pilot.select_absmax):
                logits[mode],_,caches[mode]=final.cached.run_mode(source,specs,e['prompt_ids'],None,layers=False)
    if len({id(v) for v in caches.values()})!=len(caches):raise RuntimeError('non-independent caches')
    for t,target in enumerate(e['targets'][:FINAL_TARGETS]):
        if t:
            for mode,specs in modes.items():
                if mode=='FP':logits[mode],_,caches[mode]=final.cached.run_fp(source,[e['targets'][t-1]],caches[mode],layers=False)
                else:
                    with patch.object(row,'select_rows',pilot.select_absmax):
                        logits[mode],_,caches[mode]=final.cached.run_mode(source,specs,[e['targets'][t-1]],caches[mode],layers=False)
        for mode in modes:scores[mode].add(source.torch,logits[mode],logits['FP'],[target])
    return dict(index=position,source_index=e['index'],targets=32,scores={m:s.raw() for m,s in scores.items()})


def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    progress('RESUME_AFTER_DATA_SPLIT_FAILED');progress('DATA_SPLIT_REVISED fresh=10 selection=4 final=6')
    progress('UNIT_TEST_START')
    tests=[sys.executable,'-m','unittest','tests.test_absmax_cross_layer_calibration',
        'tests.test_absmax_joint_stability','tests.test_absmax_joint_calibration',
        'tests.test_pilot_absmax_calibration','tests.test_k_selector_fidelity_384',
        'tests.test_k_selector_compare','tests.test_finalize_downproj_ptq']
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/tmp/absmax-cross-layer')
    with (OUT/'unit_test.log').open('w') as log:tested=subprocess.run(tests,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if tested.returncode:progress('UNIT_TEST_FAILED');raise RuntimeError('unit tests failed')
    progress('UNIT_TEST_DONE');progress('FROZEN_VERIFY_START')
    source,_,_,previous,_,_=final.load_source_and_policies();frozen,_=final.restore_final(source,previous)
    expected={f'model.layers.{i}.mlp.down_proj' for i in range(18)}
    if set(frozen)!=expected:raise RuntimeError('unexpected quantized modules')
    before=frozen_fingerprint(frozen);progress('FROZEN_VERIFY_DONE')
    pilot_path=ROOT/'diagnostics/absmax_joint_calibration_pilot/proposed_parameters.json'
    captured=capture_rows(frozen,before,base.sha256_file(pilot_path))
    run_key=digest(dict(frozen=before,grid=[stability.SX_JS,stability.S10_IS],capture='stability-compatible'))
    progress('CANDIDATE_GENERATION_START');pool=generate_pool(source,frozen,captured,run_key);del captured
    pool_artifact=dict(configuration_fingerprint=run_key,layers=pool);base.write_json(OUT/'candidate_pool.json',pool_artifact)
    pool_fp=base.sha256_file(OUT/'candidate_pool.json');progress('CANDIDATE_GENERATION_DONE')
    examples,_=final.validation_examples(source);excluded=previous_validation_indices()
    selection,held=deterministic_fresh_split(examples,excluded);split=make_split_artifact(selection,held,excluded)
    base.write_json(OUT/'data_split.json',split)
    progress('DATA_SPLIT_DONE selection_targets=96 final_targets=192')
    _,fp_hash=generate_fp_reference(source,selection,split['fingerprint'],run_key)
    incumbent_specs=dict(frozen)
    old_metrics=eval_selection(source,incumbent_specs,selection,fp_hash,split['fingerprint'],
                               'old_baseline',None,'old',pilot.compare.fingerprint(frozen))
    progress('OLD_ABSMAX_SELECTION_BASELINE_DONE')
    sensitivity=[];progress('SENSITIVITY_START')
    for layer in range(18):
        best=next((c for c in pool[str(layer)] if 'local_best' in c['sources']),None)
        if best is None:raise RuntimeError(f'layer {layer} lacks local_best')
        specs=apply_candidate(frozen,frozen,layer,best);started=time.monotonic()
        metrics=eval_selection(source,specs,selection,fp_hash,split['fingerprint'],'sensitivity',layer,
                               best['candidate_id'],pilot.compare.fingerprint(frozen))
        label=classify_sensitivity(old_metrics,metrics)
        sensitivity.append(dict(layer=layer,candidate_id=best['candidate_id'],classification=label,
            kl_delta=metrics['kl']-old_metrics['kl'],kl_relative=(metrics['kl']-old_metrics['kl'])/old_metrics['kl'],
            nmse_delta=metrics['nmse']-old_metrics['nmse'],top1_delta=metrics['top1']-old_metrics['top1'],
            cosine_delta=metrics['cosine']-old_metrics['cosine'],ppl_delta=metrics['ppl']-old_metrics['ppl']))
        progress(f'SENSITIVITY {layer+1}/18 DONE layer={layer} batch_sec={time.monotonic()-started:.3f} elapsed_sec={time.monotonic()-START:.3f}')
    beneficial=sorted((r for r in sensitivity if r['classification']=='BENEFICIAL'),key=lambda r:(r['kl_relative'],r['layer']))
    shortlist=[r['layer'] for r in beneficial[:MAX_SHORTLIST]]
    base.write_csv(OUT/'sensitivity.csv',sensitivity)
    base.write_json(OUT/'sensitivity_summary.json',dict(counts={k:sum(r['classification']==k for r in sensitivity)
        for k in ('BENEFICIAL','UNSAFE','NON_BENEFICIAL')},shortlist=shortlist,ranking=[r['layer'] for r in beneficial]))
    progress(f'SENSITIVITY_DONE shortlist={len(shortlist)}')
    coord=[];incumbent_specs=dict(frozen);incumbent_metrics=old_metrics
    selected_source={layer:'old' for layer in range(18)}
    for pass_number in range(1,COORDINATE_PASSES+1):
        progress(f'COORD_PASS_{pass_number}_START')
        for layer in shortlist:
            name=f'model.layers.{layer}.mlp.down_proj';current_key=pilot.compare.fingerprint(incumbent_specs)
            options=[]
            for candidate in pool[str(layer)]:
                trial=apply_candidate(incumbent_specs,frozen,layer,candidate)
                if pilot.compare.fingerprint(trial)==current_key:continue
                started=time.monotonic();metrics=eval_selection(source,trial,selection,fp_hash,split['fingerprint'],
                    f'coordinate_pass_{pass_number}',layer,candidate['candidate_id'],current_key)
                options.append(dict(candidate_id=candidate['candidate_id'],candidate=candidate,metrics=metrics,specs=trial))
                coord.append(dict(pass_number=pass_number,layer=layer,candidate_id=candidate['candidate_id'],
                    sources='+'.join(candidate['sources']),accepted=False,kl=metrics['kl'],nmse=metrics['nmse'],
                    top1=metrics['top1'],cosine=metrics['cosine'],incumbent_kl=incumbent_metrics['kl']))
                progress(f'COORD_PASS_{pass_number} layer={layer} candidate={candidate["candidate_id"]} DONE batch_sec={time.monotonic()-started:.3f} elapsed_sec={time.monotonic()-START:.3f}')
            choice=choose_coordinate(incumbent_metrics,incumbent_specs,options,frozen)
            if choice:
                incumbent_specs=choice['specs'];incumbent_metrics=choice['metrics']
                selected_source[layer]='+'.join(choice['candidate']['sources'])
                for record in reversed(coord):
                    if record['pass_number']==pass_number and record['layer']==layer and record['candidate_id']==choice['candidate_id']:
                        record['accepted']=True;break
            state=dict(pass_number=pass_number,layer=layer,configuration_fingerprint=pilot.compare.fingerprint(incumbent_specs),
                       metrics=incumbent_metrics,selected_source=selected_source)
            base.write_json(WORK/f'coordinate_state_p{pass_number}_l{layer:02d}.json',state)
        progress(f'COORD_PASS_{pass_number}_DONE changed_layers={changed_count(incumbent_specs,frozen)}')
    base.write_csv(OUT/'coordinate_search.csv',coord)
    base.write_json(OUT/'coordinate_search_summary.json',dict(shortlist=shortlist,passes=2,
        accepted_updates=sum(r['accepted'] for r in coord),changed_layers=changed_count(incumbent_specs,frozen),
        selection_metrics=incumbent_metrics))
    selected={}
    for layer in range(18):
        name=f'model.layers.{layer}.mlp.down_proj';spec=incumbent_specs[name]
        selected[name]=dict(layer=layer,source=selected_source[layer],sx=spec['sx'],s10=spec['s10'],
            multiplier=spec['params']['multiplier'].tolist(),shift=spec['params']['shift'].tolist())
    base.write_json(OUT/'selected_configuration.json',dict(changed_layers=changed_count(incumbent_specs,frozen),layers=selected))
    stored=json.loads((ROOT/'diagnostics/absmax_joint_stability/proposed_parameters.json').read_text())
    stability_specs={n:dict(frozen[n],sx=float(v['sx_base']),s10=float(v['s10']),
        params=params_for(frozen[n],v['sx_base'],v['s10'],v['multiplier'],v['shift'])) for n,v in stored.items()}
    modes=dict(FP=None,old_absmax=frozen,stability_joint_absmax=stability_specs,cross_layer_absmax=incumbent_specs)
    progress('FINAL_VALIDATION_START');records=[];last=time.monotonic()
    for number,(position,e) in enumerate(zip([x['index'] for x in held],held),1):
        checkpoint=WORK/f'final_{number:02d}.json';key=digest(dict(run=run_key,split=split['fingerprint'],
            final_configuration=pilot.compare.fingerprint(incumbent_specs),source=e['index']))
        if checkpoint_valid(checkpoint,key):record=json.loads(checkpoint.read_text())['record']
        else:
            record=final_conversation(source,modes,e,position)
            base.write_json(checkpoint,dict(checkpoint_key=key,record=record))
        records.append(record);now=time.monotonic()
        progress(f'FINAL_BATCH {number}/6 DONE targets=32 total_targets={number*32}/192 batch_sec={now-last:.3f} elapsed_sec={now-START:.3f}');last=now
    aggregate=[dict(mode=m,**pilot.response.merged([r['scores'][m] for r in records])) for m in modes]
    if any(m['tokens']!=192 for m in aggregate):raise RuntimeError('expected 192 held-out targets')
    per=[]
    for r in records:
        item=dict(index=r['index'],source_index=r['source_index'],targets=32)
        for mode in modes:
            metric=pilot.response.merged([r['scores'][mode]])
            item.update({f'{mode}_{k}':metric[k] for k in ('nll','kl','nmse','top1')})
        per.append(item)
    old,stable_metric,cross=aggregate[1:]
    deltas=dict(old_to_cross=stability.relative_deltas(old,cross),
                stability_to_cross=stability.relative_deltas(stable_metric,cross))
    gate=joint.e2e_gate(old,cross);dominance=(cross['kl']<old['kl'] and cross['nmse']<=old['nmse'] and cross['top1']>=old['top1'])
    final_summary=dict(modes=aggregate,deltas=deltas,gate=gate,
        classification='CROSS_LAYER_PASS' if gate['pass'] else 'CROSS_LAYER_REGRESSION',
        dominance='STRICT_IMPROVEMENT' if dominance else 'TRADEOFF')
    base.write_json(OUT/'final_validation_summary.json',final_summary);base.write_csv(OUT/'final_validation_per_conversation.csv',per)
    after=frozen_fingerprint(frozen)
    if before!=after:raise RuntimeError('frozen artifacts changed')
    verification=dict(frozen_before=before,frozen_after=after,frozen_unchanged=True,
        quantized_modules=sorted(expected),selector_identity=joint.select_absmax is pilot.select_absmax,
        local_candidate_policy_identity='absmax_joint_calibration',candidate_grid=dict(sx=stability.SX_JS,s10=stability.S10_IS),
        candidate_pool_fingerprint=pool_fp,previous_validation_exclusion_source_indices=sorted(excluded),
        available_fresh_conversations=10,selection_source_indices=[e['index'] for e in selection],
        final_heldout_source_indices=[e['index'] for e in held],selection_conversations=4,selection_targets=96,
        final_conversations=6,final_targets=192,previous_validation_overlap=False,selection_final_overlap=False,
        candidate_generation_selection_overlap=False,candidate_generation_final_overlap=False,
        independent_kv_caches=True,effective_shifts_feasible=all(
            0<=c['effective_shift_min']<=c['effective_shift_max']<=31
            for candidates in pool.values() for c in candidates),
        coordinate_passes=2,final_heldout_not_used_in_search=True,test_command=tests,test_return_code=tested.returncode,
        data_split_fingerprint=split['fingerprint'],selection_fp_reference_fingerprint=fp_hash,elapsed_sec=time.monotonic()-START)
    base.write_json(OUT/'verification.json',verification)
    unsafe=[r['layer'] for r in sensitivity if r['classification']=='UNSAFE'];nonbenef=[r['layer'] for r in sensitivity if r['classification']=='NON_BENEFICIAL']
    local_but_e2e_bad=[r['layer'] for r in sensitivity if r['classification']!='BENEFICIAL']
    lines=['# Absmax cross-layer E2E calibration','',
        'The final held-out evaluation is smaller than originally planned because only 10 genuinely fresh validation conversations remained after excluding all prior evaluation source indices.',
        'It is an independent held-out exploratory validation, not a definitive model evaluation; no statistical significance or optimality is claimed.','',
        f'Candidate generation used the fixed 45-pair joint grid. The 96-target selection set and 192-target held-out set are disjoint and exclude prior validation source indices.',
        f'Individually E2E-beneficial layers: {[r["layer"] for r in beneficial]}; unsafe: {unsafe}; non-beneficial: {nonbenef}.',
        f'Shortlist: {shortlist}. Final configuration changes {changed_count(incumbent_specs,frozen)} layers.',
        f'Local NMSE improved while individual E2E KL failed to improve for layers: {local_but_e2e_bad}. This is consistent with cross-layer interaction/error cancellation being important, but does not prove it.','',
        '| Held-out mode | NLL | PPL | KL | NMSE | Cosine | Top1 | FP top1 in top5 | Top5 overlap |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in aggregate:lines.append('| '+m['mode']+' | '+' | '.join(f'{m[k]:.9g}' for k in ('nll','ppl','kl','nmse','cosine','top1','in5','overlap'))+' |')
    lines += ['',f'Old to cross-layer deltas: `{json.dumps(deltas["old_to_cross"])}`.',
              f'Stability-joint to cross-layer deltas: `{json.dumps(deltas["stability_to_cross"])}`.',
              f'Gate: `{json.dumps(gate)}`; classification **{final_summary["classification"]}**, relation to old **{final_summary["dominance"]}**.',
              'Held-out metrics were computed only after selection froze and did not feed back into the search.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    progress('ARTIFACT_WRITE_DONE');progress('FINAL_VERIFY_DONE');progress('RUN_COMPLETE')


if __name__=='__main__':
    try:main()
    except BaseException:
        progress('RUN_FAILED');raise
