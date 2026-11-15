"""Small-batch full-scale GPALU calibration with a Down-aware objective."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import run_gpalu_pot_fusion_pilot as pot

fusion,mixed,gelu_base=pot.fusion,pot.mixed,pot.gelu_base
base,cal,final,response,Profile=pot.base,pot.cal,pot.final,pot.response,pot.Profile
OUT=ROOT/"diagnostics/gpalu_fullscale_downaware_pilot";WORK=OUT/"work"
LUT_DIR=OUT/"gelu_tables";GATE_QB=OUT/"gate_qparams";DOWN_QB=OUT/"down_qparams"
MIXED=ROOT/"diagnostics/mixed_linear_gelu_calibration"
FUSION=ROOT/"diagnostics/fusion_aware_gate_s10_pilot"
POT=ROOT/"diagnostics/gpalu_pot_fusion_pilot"
PROGRESS=ROOT/"gpalu_fullscale_downaware_progress.txt"
REQUIRED=("2d77cf52907d8b1f910e26a1952c69248090190f","f92fafb2db225165a2eb9064b5073e6f212fac54","cbc545ddc00cf3ed697217de0748f9a396f479dc")
PERCENTILES=(99.,99.5,99.9,99.95,99.99,99.995,100.)
J_VALUES=tuple(range(-4,5));GROUPS=("prefill","decode")
MODES=("FP_FULL","FUSION_AWARE_FLOAT_PRODUCT","STATIC_POT_KG","FULL_SCALE_LOCAL_HW","FULL_SCALE_DOWN_AWARE_HW")
DOWN_SHORTLIST=16
START=time.monotonic()


def progress(message):
    with PROGRESS.open("a") as f:f.write(message+"\n");f.flush()


def sha(path):return base.sha256_file(Path(path))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def write_csv(path,rows):base.write_csv(path,rows);path.write_bytes(path.read_bytes().replace(b"\r\n",b"\n"))
def balanced(values):return float(np.mean([v for v in values if v is not None]))


def group_nmse(value,reference,labels):
    return {g:(mixed.nmse(value[labels==g],reference[labels==g]) if np.any(labels==g) else None) for g in GROUPS}


def approximate_scalar(alpha):
    p=Profile().approximate(np.array([alpha],np.float64))
    if p["status"][0]!="ok":raise RuntimeError(f"invalid GPALU ratio: {alpha} {p['status'][0]}")
    m,s=int(p["multiplier"][0]),int(p["shift"][0])
    if not 0<=m<=65535 or not 0<=s<=31 or m==0:raise RuntimeError("invalid GPALU M/S")
    return {"M_G":m,"S_G":s,"alpha_hw":m/(2.**s),"alpha_relative_error":float(p["relative_error"][0])}


def fullscale_codes(product,alpha,multiplier,shift):
    product=np.asarray(product,np.int64)
    ideal_pre=np.rint(product.astype(np.float64)*alpha).astype(np.int64)
    hw_pre=Profile().apply(product,np.array([multiplier]),np.array([shift]),saturate=False)
    ideal=np.clip(ideal_pre,-128,127).astype(np.int8);hw=np.clip(hw_pre,-128,127).astype(np.int8)
    return ideal,hw,{"ideal_preclip_min":int(ideal_pre.min()),"ideal_preclip_max":int(ideal_pre.max()),
        "hw_preclip_min":int(hw_pre.min()),"hw_preclip_max":int(hw_pre.max()),
        "ideal_clip_count":int(((ideal_pre< -128)|(ideal_pre>127)).sum()),"ideal_clip_rate":float(np.mean((ideal_pre< -128)|(ideal_pre>127))),
        "hw_clip_count":int(((hw_pre< -128)|(hw_pre>127)).sum()),"hw_clip_rate":float(np.mean((hw_pre< -128)|(hw_pre>127))),
        "positive_clip_rate":float(np.mean(hw_pre>127)),"negative_clip_rate":float(np.mean(hw_pre< -128))}


def effective_scale(s_product,multiplier,shift):return float(s_product*(2.**shift)/multiplier)


def exact_dot(q,wq):
    out=np.asarray(q,np.float64)@np.asarray(wq,np.float64).T
    if not np.array_equal(out,np.rint(out)):raise RuntimeError("noninteger exact dot")
    return out.astype(np.int64)


def direct_down(q_h,s_h,down_spec):
    params=Profile().approximate(s_h*down_spec["sw"]/down_spec["sout"])
    if np.any(params["status"]!="ok"):raise RuntimeError("invalid Down profile")
    acc=exact_dot(q_h,down_spec["wq"])
    raw=Profile().apply(acc,params["multiplier"],params["shift"],saturate=False)
    codes=np.clip(raw,-128,127).astype(np.int8)
    return codes.astype(np.float64)*down_spec["sout"],params,float(np.mean((raw< -128)|(raw>127)))


def scale_anchors(h_fp,h_pre,down_sx,static_scale):
    coarse=[]
    for source,values in (("fp",h_fp),("pre",h_pre)):
        absolute=np.abs(values).reshape(-1)
        coarse.extend((float(np.percentile(absolute,p))/127.,source,p) for p in PERCENTILES)
    coarse.extend(((float(down_sx),"old_down",None),(float(static_scale),"static_pot",None)))
    coarse=[item for item in coarse if np.isfinite(item[0]) and item[0]>0]
    expanded=[]
    for anchor,source,p in coarse:
        expanded.extend((anchor*2.**(j/16.),source,p,j) for j in J_VALUES)
    expanded.sort(key=lambda x:x[0])
    result=[]
    for item in expanded:
        if not result or not np.isclose(item[0],result[-1][0],rtol=1e-12,atol=0):result.append(item)
    return result


def cancellation_warning(selected,best_local):return selected["local_hw_nmse_balanced"]>2.*best_local["local_hw_nmse_balanced"]


def down_key(c):return (c["down_nmse_balanced"],c["down_nmse_worst"],c["local_hw_nmse_balanced"],c["hw_clip_rate"],c["alpha_relative_error"],c["down_output_saturation_rate"],abs(c["anchor_log2_offset"]),c["s_h_target"])
def local_key(c):return (c["local_hw_nmse_balanced"],c["hw_clip_rate"],c["alpha_relative_error"],c["s_h_target"])
def ideal_key(c):return (c["local_ideal_nmse_balanced"],c["ideal_clip_rate"],c["s_h_target"])


def s10_candidates(current,original,static):
    rows=[{"s10_j":j,"s10":current*2.**(j/8.),"source":"fusion_grid"} for j in range(-4,5)]
    for value,source in ((original,"original_mixed"),(static,"static_pot")):
        if not any(item["s10"]==value for item in rows):rows.append({"s10_j":source,"s10":value,"source":source})
    return rows


def evaluate_layer(layer,x,labels,gate_base,gate_current,up_spec,down_spec,old_lut,current_lut,static_info):
    g_fp=x.astype(np.float64)@gate_base["weight"].astype(np.float64).T
    u_fp=x.astype(np.float64)@up_spec["weight"].astype(np.float64).T
    u_hat,u_details=mixed.quantized_linear_rows(x,up_spec,return_codes=True);q_u=u_details["output_codes"].astype(np.int16)
    h_fp=gelu_base.gelu(g_fp)*u_fp;y_down_fp=h_fp@down_spec["weight"].astype(np.float64).T
    all_candidates=[];payload={};local_probes=[]
    for sitem in s10_candidates(gate_current["sout"],gate_base["sout"],static_info["s10_selected"]):
        s10=sitem["s10"];params=Profile().approximate(gate_base["sx"]*gate_base["sw"]/s10)
        if np.any(params["status"]!="ok"):raise RuntimeError("invalid Gate profile")
        gate_spec=dict(gate_base,sout=s10,params=params)
        g_hat,g_details=mixed.quantized_linear_rows(x,gate_spec,return_codes=True);q10=g_details["output_codes"]
        inner,_=fusion.inner_sact(g_fp,q10,labels,s10)
        q_a=inner["lut"][gelu_base.address(q10)].astype(np.int16);a_hat=q_a.astype(np.float64)*inner["s_act"]
        h_pre=a_hat*u_hat;s_product=inner["s_act"]*up_spec["sout"]
        p16=(q_a.astype(np.int32)*q_u.astype(np.int32));
        if np.any((p16< -32768)|(p16>32767)):raise RuntimeError("raw INT16 overflow")
        raw_stats={"raw_product_min":int(p16.min()),"raw_product_max":int(p16.max()),"raw_product_absmax":int(np.abs(p16).max())}
        probes=[]
        for target,source,percentile,offset in scale_anchors(h_fp,h_pre,down_spec["sx"],static_info["output_scale"]):
            alpha=s_product/target;rep=approximate_scalar(alpha);effective=effective_scale(s_product,rep["M_G"],rep["S_G"])
            ideal,hw,clip=fullscale_codes(p16,alpha,rep["M_G"],rep["S_G"])
            h_ideal=ideal.astype(np.float64)*target;h_hw=hw.astype(np.float64)*effective
            ideal_group=group_nmse(h_ideal,h_fp,labels);hw_group=group_nmse(h_hw,h_fp,labels)
            ms_inc=group_nmse(h_hw,h_ideal,labels)
            probes.append({"layer":layer,"s10_j":sitem["s10_j"],"s10":s10,"s10_source":sitem["source"],"s_act":inner["s_act"],"gelu_percentile":inner["retained_percentile"],
                "s_product":s_product,"s_h_target":target,"s_h_effective":effective,"M_G":rep["M_G"],"S_G":rep["S_G"],"alpha_target":alpha,"alpha_hw":rep["alpha_hw"],
                "alpha_relative_error":rep["alpha_relative_error"],"s_h_relative_error":(effective-target)/target,"local_ideal_nmse_balanced":balanced(ideal_group.values()),
                "local_hw_nmse_balanced":balanced(hw_group.values()),"MS_representation_incremental_nmse":balanced(ms_inc.values()),
                **raw_stats,**clip,"anchor_source":source,"anchor_percentile":percentile,"anchor_log2_offset":offset,"q_h_hw":hw,"q_h_ideal":ideal,"params_gate":params,"lut":inner["lut"]})
        # Down evaluation is bounded to the locally strongest scales plus mandatory architectural anchors.
        ranked=sorted(probes,key=local_key);mandatory=[min(probes,key=ideal_key)]
        mandatory += [min(probes,key=lambda c:abs(math.log2(c["s_h_target"]/anchor))) for anchor in (down_spec["sx"],static_info["output_scale"])]
        shortlist=[]
        for c in mandatory+ranked:
            if not any(x["s_h_target"]==c["s_h_target"] for x in shortlist):shortlist.append(c)
            if len(shortlist)>=DOWN_SHORTLIST:break
        for c in shortlist:
            y_hat,down_params,down_sat=direct_down(c["q_h_hw"],c["s_h_effective"],down_spec)
            down_group=group_nmse(y_hat,y_down_fp,labels)
            c.update(down_nmse_prefill=down_group["prefill"],down_nmse_decode=down_group["decode"],down_nmse_balanced=balanced(down_group.values()),
                     down_nmse_worst=max(v for v in down_group.values() if v is not None),down_output_saturation_rate=down_sat,down_params=down_params,
                     scale_probe_count=len(probes),selected_for_s10=False,selected_final=False,error_cancellation_warning=False)
            all_candidates.append(c)
        best_down=min(shortlist,key=down_key);best_down["selected_for_s10"]=True
        local_probes.extend(probes)
    selected=min((c for c in all_candidates if c["selected_for_s10"]),key=down_key);selected["selected_final"]=True
    best_local=min(local_probes,key=local_key);best_ideal=min(local_probes,key=ideal_key)
    # Ensure local-HW configuration has complete Down metadata for E2E.
    local_match=next((c for c in all_candidates if c["s10"]==best_local["s10"] and c["s_h_target"]==best_local["s_h_target"]),None)
    if local_match is None:
        y_hat,dp,ds=direct_down(best_local["q_h_hw"],best_local["s_h_effective"],down_spec);dg=group_nmse(y_hat,y_down_fp,labels)
        best_local.update(down_nmse_prefill=dg["prefill"],down_nmse_decode=dg["decode"],down_nmse_balanced=balanced(dg.values()),down_nmse_worst=max(dg.values()),down_output_saturation_rate=ds,down_params=dp)
        local_match=best_local
    selected["error_cancellation_warning"]=cancellation_warning(selected,best_local)
    for c in all_candidates:
        for key in ("q_h_hw","q_h_ideal","params_gate","lut","down_params"):c.pop(key,None)
    return selected,local_match,best_ideal,all_candidates


def compact_payload(candidate,gate_base,down_base,lut):
    gate_params=Profile().approximate(gate_base["sx"]*gate_base["sw"]/candidate["s10"])
    down_params=Profile().approximate(candidate["s_h_effective"]*down_base["sw"]/down_base["sout"])
    return {"gate_spec":dict(gate_base,sout=candidate["s10"],params=gate_params),"lut":{"s10":candidate["s10"],"s_act":candidate["s_act"],"lut":lut},
            "M_G":candidate["M_G"],"S_G":candidate["S_G"],"s_h_effective":candidate["s_h_effective"],"down_spec":dict(down_base,params=down_params)}


@contextlib.contextmanager
def patch_fullscale(source,specs,luts,gpalu_params,down_specs):
    originals=[]
    with mixed.patch_mixed(source,specs,luts):
        try:
            for layer,block in enumerate(source.model.model.layers):
                mlp=block.mlp;originals.append((mlp,mlp.forward));item=luts[layer];gp=gpalu_params[layer];up_scale=specs[f"model.layers.{layer}.mlp.up_proj"]["sout"];down=down_specs[layer]
                def forward(x,mlp=mlp,item=item,gp=gp,up_scale=up_scale,down=down):
                    a=mlp.act_fn(mlp.gate_proj(x));u=mlp.up_proj(x)
                    qa=np.clip(np.rint(a.detach().cpu().numpy()/item["s_act"]),-127,127).astype(np.int16);qu=np.clip(np.rint(u.detach().cpu().numpy()/up_scale),-128,127).astype(np.int16)
                    product=qa.astype(np.int32)*qu.astype(np.int32);_,qh,_=fullscale_codes(product,gp["alpha_target"],gp["M_G"],gp["S_G"])
                    y,_,_=direct_down(qh.reshape(-1,qh.shape[-1]),gp["s_h_effective"],down);y=y.reshape(*qh.shape[:-1],y.shape[-1])
                    return x.new_tensor(y.astype(np.float32))
                mlp.forward=forward
            yield
        finally:
            for module,forward in originals:module.forward=forward


def run_fullscale(source,specs,luts,gpalu_params,down_specs,ids,cache):
    with patch_fullscale(source,specs,luts,gpalu_params,down_specs):
        hidden,states,cache=response.forward_hidden(source,ids,[len(ids)-1],{},cache,True,False)
        with source.torch.inference_mode():logits=source.model.lm_head(hidden)
    return logits,states,cache


def evaluate_sequence(source,local_cfg,down_cfg,example,stored):
    modes=("FULL_SCALE_LOCAL_HW","FULL_SCALE_DOWN_AWARE_HW");scores={m:response.Scores() for m in modes};caches={m:None for m in modes}
    for t,target in enumerate(example["targets"][:24]):
        ids=example["prompt_ids"] if t==0 else [example["targets"][t-1]]
        logits={}
        for mode,cfg in ((modes[0],local_cfg),(modes[1],down_cfg)):
            logits[mode],_,caches[mode]=run_fullscale(source,cfg["specs"],cfg["luts"],cfg["gpalu"],cfg["down"],ids,caches[mode])
        if caches[modes[0]] is caches[modes[1]]:raise RuntimeError("KV cache alias")
        fp_logits=stored["fp_logits"][t] if "fp_logits" in stored else None
        # Stored raw metrics do not contain logits, so obtain aligned FP once for scoring both new modes.
        if t==0:fp_cache=None
        fp_logits,_,fp_cache=cal.run_fp(source,ids,fp_cache,layers=False)
        for mode in modes:scores[mode].add(source.torch,logits[mode],fp_logits,[target])
    return {mode:scores[mode].raw() for mode in modes}


def delta(old,new):
    d={k:{"absolute":new[k]-old[k],"relative":(new[k]-old[k])/old[k]} for k in ("kl","nmse","ppl")};d["top1"]={"absolute_pp":100*(new["top1"]-old["top1"])};return d


def main():
    PROGRESS.write_text("");progress("RUN_START")
    for p in (OUT,WORK,LUT_DIR,GATE_QB,DOWN_QB):p.mkdir(parents=True,exist_ok=True)
    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    if branch!="lut_cali":raise RuntimeError("wrong branch")
    for commit in REQUIRED:
        if subprocess.run(["git","merge-base","--is-ancestor",commit,"HEAD"]).returncode:raise RuntimeError(f"missing commit {commit}")
    progress("GIT_CHECK_DONE")
    baseline_paths=[MIXED/"linear_parameters.json",MIXED/"capture_manifest.json",FUSION/"gate_parameters.json",POT/"gpalu_parameters.json",POT/"e2e_dataset_manifest.json"]
    baseline_hashes={str(p.relative_to(ROOT)):sha(p) for p in baseline_paths};base.write_json(OUT/"baseline_manifest.json",{"required_commits":list(REQUIRED),"artifact_hashes":baseline_hashes,"status":"READ_ONLY_BASELINES"});progress("BASELINE_VERIFY_DONE")
    tests=[sys.executable,"-m","unittest","tests.test_gpalu_fullscale_downaware_pilot","tests.test_gpalu_pot_fusion_pilot","tests.test_fusion_aware_gate_s10_pilot","tests.test_mixed_linear_gelu_calibration"]
    with (OUT/"unit_test.log").open("w") as log:code=subprocess.run(tests,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPYCACHEPREFIX="/tmp/gpalu-fullscale",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",KMP_USE_SHM="0")).returncode
    if code:progress("UNIT_TEST_FAILED");raise RuntimeError("tests failed")
    source,_,_,_,_,_=final.load_source_and_policies();historical_manifest=json.loads((mixed.ART/"manifest.json").read_text());historical=mixed.load_historical(source,historical_manifest)
    baseline_specs=fusion.load_baseline_specs(source,historical);current_specs=pot.load_current_specs(baseline_specs);current_luts=pot.load_current_luts();pot_meta=json.loads((POT/"gpalu_parameters.json").read_text())["layers"]
    capture=MIXED/"work/capture.npz";meta=MIXED/"work/capture.json"
    if not capture.exists() or sha(capture)!=json.loads(meta.read_text())["sha256"]:raise RuntimeError("capture unavailable")
    with np.load(capture) as z:captured={k:z[k] for k in z.files}
    progress("CAPTURE_REUSE_DONE")
    inventory=mixed.expected_inventory();candidate_rows=[];selection_rows=[];local_rows=[];local_payloads={};down_payloads={};progress("SEARCH_START")
    for layer in range(18):
        names={s:f"model.layers.{layer}.mlp.{s}" for s in ("gate_proj","up_proj","down_proj")};idx=inventory.index(names["gate_proj"])
        x=np.concatenate([captured[f"op{idx:04d}_{g}"] for g in GROUPS]);labels=np.concatenate([[g]*len(captured[f"op{idx:04d}_{g}"]) for g in GROUPS])
        selected,local,ideal,rows=evaluate_layer(layer,x,labels,baseline_specs[names["gate_proj"]],current_specs[names["gate_proj"]],baseline_specs[names["up_proj"]],baseline_specs[names["down_proj"]],fusion.load_baseline_luts(baseline_specs)[layer],current_luts[layer],pot_meta[layer])
        candidate_rows.extend(rows);selected["error_cancellation_warning"]=cancellation_warning(selected,local)
        selection_rows.append({k:v for k,v in selected.items() if k not in ("q_h_hw","q_h_ideal","params_gate","lut","down_params")})
        local_rows.append({"layer":layer,"pre_gpalu_float_nmse":next(r["current_float_pre_gpalu_nmse"] for r in csv_rows(POT/"layer_selection.csv") if int(r["layer"])==layer),"static_pot_nmse":pot_meta[layer]["post_gpalu_nmse"],"fullscale_ideal_local_nmse":ideal["local_ideal_nmse_balanced"],"fullscale_local_hw_nmse":local["local_hw_nmse_balanced"],"fullscale_downaware_hw_nmse":selected["local_hw_nmse_balanced"],"downaware_down_nmse":selected["down_nmse_balanced"]})
        # Recreate selected LUT and derived qparams without retaining tensors in CSV.
        for chosen,target in ((local,"local"),(selected,"down")):
            gp=Profile().approximate(baseline_specs[names["gate_proj"]]["sx"]*baseline_specs[names["gate_proj"]]["sw"]/chosen["s10"])
            spec=dict(baseline_specs[names["gate_proj"]],sout=chosen["s10"],params=gp)
            gfp=x.astype(np.float64)@spec["weight"].astype(np.float64).T;_,gd=mixed.quantized_linear_rows(x,spec,return_codes=True);inner,_=fusion.inner_sact(gfp,gd["output_codes"],labels,chosen["s10"])
            payload=compact_payload(chosen,baseline_specs[names["gate_proj"]],baseline_specs[names["down_proj"]],inner["lut"])
            (local_payloads if target=="local" else down_payloads)[layer]=payload
        progress(f"LAYER {layer+1}/18 DONE")
    write_csv(OUT/"candidate_summary.csv",candidate_rows);write_csv(OUT/"layer_selection.csv",selection_rows);write_csv(OUT/"local_baseline_comparison.csv",local_rows);progress("SEARCH_DONE")
    # Export selected parameter payloads.
    gpalu_json=[];down_json=[];scale_rows=[];gate_json=[];lut_hashes=[];gate_hashes=[];down_hashes=[]
    for layer,row in enumerate(selection_rows):
        payload=down_payloads[layer];gp=payload["gate_spec"]["params"];dp=payload["down_spec"]["params"]
        gpath=GATE_QB/f"layer_{layer:02d}_gate_qb.bin";gpath.write_bytes(Profile().pack(gp["multiplier"],gp["shift"]).astype("<u4").tobytes());gate_hashes.append(sha(gpath))
        dpath=DOWN_QB/f"layer_{layer:02d}_down_qb.bin";dpath.write_bytes(Profile().pack(dp["multiplier"],dp["shift"]).astype("<u4").tobytes());down_hashes.append(sha(dpath))
        lpath=LUT_DIR/f"layer_{layer:02d}_gelu.bin";lpath.write_bytes(payload["lut"]["lut"].tobytes());lut_hashes.append(sha(lpath));base.write_json(LUT_DIR/f"layer_{layer:02d}_gelu.json",{"layer":layer,"s10":row["s10"],"s_act":row["s_act"],"sha256":lut_hashes[-1]})
        gpalu_json.append({"layer":layer,"s10_selected":row["s10"],"s_act_selected":row["s_act"],"s8_up_fixed":baseline_specs[f"model.layers.{layer}.mlp.up_proj"]["sout"],"s_product":row["s_product"],"s_h_target":row["s_h_target"],"s_h_effective":row["s_h_effective"],"M_G":row["M_G"],"S_G":row["S_G"],"ZP_G":0,"alpha_target":row["alpha_target"],"alpha_hw":row["alpha_hw"],"alpha_relative_error":row["alpha_relative_error"],"local_ideal_nmse":row["local_ideal_nmse_balanced"],"local_hw_nmse":row["local_hw_nmse_balanced"],"MS_representation_incremental_nmse":row["MS_representation_incremental_nmse"],"GPALU_clip_rate":row["hw_clip_rate"],"old_sX_down_base":baseline_specs[f"model.layers.{layer}.mlp.down_proj"]["sx"],"output_to_old_down_scale_ratio":row["s_h_effective"]/baseline_specs[f"model.layers.{layer}.mlp.down_proj"]["sx"],"consumer_aware_down_nmse":row["down_nmse_balanced"]})
        down_json.append({"layer":layer,"input_scale":row["s_h_effective"],"output_scale":baseline_specs[f"model.layers.{layer}.mlp.down_proj"]["sout"],"qparam_binary_path":str(dpath.relative_to(ROOT)),"M_min":int(dp["multiplier"].min()),"M_max":int(dp["multiplier"].max()),"S_min":int(dp["shift"].min()),"S_max":int(dp["shift"].max()),"output_saturation_rate":row["down_output_saturation_rate"],"qparam_sha256":down_hashes[-1]})
        ratio=gpalu_json[-1]["output_to_old_down_scale_ratio"];scale_rows.append({"layer":layer,"s_h_target":row["s_h_target"],"s_h_effective":row["s_h_effective"],"relative_error":row["s_h_relative_error"],"alpha_relative_error":row["alpha_relative_error"],"old_sX_down_base":gpalu_json[-1]["old_sX_down_base"],"ratio":ratio,"log2_ratio":math.log2(ratio),"nearest_old_pot_k":int(np.rint(math.log2(ratio))),"within_old_pot_family":bool(np.isclose(ratio,2.**np.rint(math.log2(ratio)),rtol=1e-6))})
        gate_json.append({"layer":layer,"s10":row["s10"],"qparam_binary_path":str(gpath.relative_to(ROOT)),"qparam_sha256":gate_hashes[-1],"lut_path":str(lpath.relative_to(ROOT)),"lut_sha256":lut_hashes[-1]})
    base.write_json(OUT/"gpalu_parameters.json",{"layers":gpalu_json});base.write_json(OUT/"down_requant_parameters.json",{"layers":down_json});base.write_json(OUT/"gate_parameters.json",{"layers":gate_json});write_csv(OUT/"scale_representation_summary.csv",scale_rows);write_csv(OUT/"down_scale_alignment.csv",scale_rows);progress("MS_REPRESENTATION_DONE");progress("DOWN_AWARE_DONE")
    max_p=max(max(abs(int(r["raw_product_min"])) if "raw_product_min" in r else 0,abs(int(r["raw_product_max"])) if "raw_product_max" in r else 0) for r in candidate_rows) if candidate_rows else 16256
    max_m=max(r["M_G"] for r in selection_rows);max_product=max_p*max_m;observed_bits=1+math.ceil(math.log2(max_product+1));theoretical_max=32768*65535;theoretical_bits=1+math.ceil(math.log2(theoretical_max+1))
    width={"raw_product_bits":16,"multiplier_bits":16,"theoretical_max_abs_product_times_multiplier":theoretical_max,"theoretical_minimum_signed_intermediate_bits":theoretical_bits,"observed_max_abs_p16":max_p,"observed_max_M_G":max_m,"observed_max_abs_product_times_multiplier":max_product,"observed_minimum_signed_intermediate_bits":observed_bits}
    base.write_json(OUT/"hardware_width_summary.json",width);progress("WIDTH_ANALYSIS_DONE")
    # E2E: reuse the already aligned FP/float/static modes; run only the two new configurations.
    manifest=json.loads((POT/"e2e_dataset_manifest.json").read_text());base.write_json(OUT/"e2e_dataset_manifest.json",manifest);examples=mixed.e2e_examples(source)
    def cfg(payloads):
        specs=dict(current_specs);luts={};gpdict={};down={}
        for layer,p in payloads.items():specs[f"model.layers.{layer}.mlp.gate_proj"]=p["gate_spec"];luts[layer]=p["lut"];gpdict[layer]={"alpha_target":p["M_G"]/(2.**p["S_G"]),"M_G":p["M_G"],"S_G":p["S_G"],"s_h_effective":p["s_h_effective"]};down[layer]=p["down_spec"]
        return {"specs":specs,"luts":luts,"gpalu":gpdict,"down":down}
    local_cfg,down_cfg=cfg(local_payloads),cfg(down_payloads);run_id=digest({"baseline":baseline_hashes,"selection":selection_rows,"data":manifest});records=[];progress("E2E_START")
    for i,example in enumerate(examples):
        prior=json.loads((POT/"work"/f"e2e_{i:02d}.json").read_text())["value"];path=WORK/f"e2e_{i:02d}.json";identity=digest({"run":run_id,"example":mixed.identity(example)});cached=json.loads(path.read_text()) if path.exists() else None
        if cached and cached.get("identity")!=identity:raise RuntimeError("checkpoint mismatch")
        new=cached["value"] if cached else evaluate_sequence(source,local_cfg,down_cfg,example,prior)
        if not cached:base.write_json(path,{"identity":identity,"value":new})
        records.append({"FP_FULL":prior["FP_FULL"],"FUSION_AWARE_FLOAT_PRODUCT":prior["CURRENT_FUSION_AWARE_FLOAT_PRODUCT"],"STATIC_POT_KG":prior["JOINT_POST_GPALU_AWARE"],**new});progress(f"SEQUENCE {i+1}/8 DONE")
    metrics={m:response.merged([r[m] for r in records]) for m in MODES};comparisons={"static_to_local":delta(metrics[MODES[2]],metrics[MODES[3]]),"local_to_downaware":delta(metrics[MODES[3]],metrics[MODES[4]]),"float_to_downaware":delta(metrics[MODES[1]],metrics[MODES[4]])}
    recovery=sum(metrics[MODES[4]][k]<metrics[MODES[2]][k] for k in ("kl","nmse"))+(metrics[MODES[4]]["top1"]>metrics[MODES[2]]["top1"]);residual=sum(metrics[MODES[4]][k]>metrics[MODES[1]][k] for k in ("kl","nmse"))+(metrics[MODES[4]]["top1"]<metrics[MODES[1]]["top1"])
    decision="PROCEED_TO_LARGE_BATCH" if recovery>=2 and residual<=1 else ("FULL_SCALE_NOT_SUPPORTED_BY_PILOT" if recovery==0 else "FULL_SCALE_PROMISING_BUT_NEEDS_REFINEMENT")
    base.write_json(OUT/"e2e_summary.json",{"modes":metrics,"comparisons":comparisons,"decision":decision,"e2e_used_for_selection":False});progress("E2E_DONE")
    if {str(p.relative_to(ROOT)):sha(p) for p in baseline_paths}!=baseline_hashes:raise RuntimeError("baseline mutation")
    verification={"branch":branch,"source_HEAD":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"required_commits_present":True,"model_SHA256":historical_manifest["gguf"]["sha256"],"baseline_artifact_hashes":baseline_hashes,"baseline_artifacts_unchanged":True,"Gate_op_count":18,"Up_op_count":18,"Down_op_count":18,"ordinary_params_changed":False,"Up_params_changed":False,"Gate_sx_changed":False,"Down_Wq_sW_s8_changed":False,"Down_MS_regenerated":True,"s10_search_performed":True,"fullscale_search_performed":True,"GPALU_multiplier_format":"UInt16","GPALU_shift_format":"UInt5","GPALU_zero_point":0,"raw_product_dtype":"signed_INT16","raw_product_overflow_count":0,"post_GPALU_dtype":"signed_INT8","dynamic_GPALU_scale":False,"direct_GPALU_to_Down":True,"second_input_quantizer":False,"candidate_down_shortlist_per_s10":DOWN_SHORTLIST,"E2E_used_for_selection":False,"large_batch_performed":False,"error_cancellation_warning_layers":[r["layer"] for r in selection_rows if r["error_cancellation_warning"]],"hardware_width_summary":width,"all_gate_qparam_hashes":gate_hashes,"all_down_qparam_hashes":down_hashes,"all_LUT_hashes":lut_hashes,"test_command":tests,"test_return_code":code,"elapsed_sec":time.monotonic()-START}
    base.write_json(OUT/"verification.json",verification)
    lines=["# Full-scale GPALU Down-aware small-batch pilot","","## 1. Scope","Static per-operation full-scale GPALU requantization and direct Down consumption were calibrated on the existing small batch.","## 2. Why POT-only failed","The prior POT-only pilot had negligible clipping but substantial scale-resolution loss.","## 3. Full-scale M_G/S_G hardware contract","GPALU used full signed INT16 raw multiplication without operand pre-shifts, followed by static UInt16 M_G / 2^UInt5 S_G, RNE, and signed INT8 saturation.","## 4. Direct GPALU -> Down scale contract","GPALU INT8 output fed Down directly. No second GPALU-output-to-Down-input quantization was performed.","## 5. Frozen parameters","Up, Gate sX/Wq/sW, Down Wq/sW/output s8, other GEMMs, weights, and runtime policies were fixed. Down M/S were regenerated because the physical input scale changed.","## 6. Search policy",f"Gate s10 used the fixed small grid. Scale probes used requested percentile anchors and log neighborhoods; {DOWN_SHORTLIST} strongest and mandatory anchor candidates per s10 received exact Down evaluation. E2E was not used for selection.","## 7. M/S representation accuracy",f"Mean alpha relative error: {np.mean([r['alpha_relative_error'] for r in selection_rows]):.9g}.","## 8. Per-layer selected parameters","| Layer | s10 | s_act | target s_h | effective s_h | M_G | S_G | Down NMSE | warning |","|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in selection_rows:lines.append(f"| {r['layer']} | {r['s10']:.9g} | {r['s_act']:.9g} | {r['s_h_target']:.9g} | {r['s_h_effective']:.9g} | {r['M_G']} | {r['S_G']} | {r['down_nmse_balanced']:.9g} | {r['error_cancellation_warning']} |")
    lines += ["## 9. GPALU clipping",f"Mean selected clip rate: {np.mean([r['hw_clip_rate'] for r in selection_rows]):.9g}.","## 10. Local GeGLU fidelity",f"Mean selected ideal/HW local NMSE: {np.mean([r['local_ideal_nmse_balanced'] for r in selection_rows]):.9g}/{np.mean([r['local_hw_nmse_balanced'] for r in selection_rows]):.9g}; mean M/S representation incremental NMSE: {np.mean([r['MS_representation_incremental_nmse'] for r in selection_rows]):.9g}.","## 11. Down-projection consumer-aware fidelity",f"Mean selected Down NMSE: {np.mean([r['down_nmse_balanced'] for r in selection_rows]):.9g}.","## 12. Error-cancellation warnings",f"Layers: {[r['layer'] for r in selection_rows if r['error_cancellation_warning']] or 'none'}.","## 13. Old Down scale vs new GPALU output scale","See `down_scale_alignment.csv`; equality was not forced.","## 14. Small E2E comparison","| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode,v in metrics.items():lines.append("| "+mode+" | "+" | ".join(f"{v[k]:.9g}" for k in ("nll","ppl","kl","nmse","mse","mae","cosine","flattened_cosine","top1","in5","overlap"))+" |")
    lines += ["### E2E deltas","| Comparison | KL rel. | NMSE rel. | PPL rel. | Top1 pp |","|---|---:|---:|---:|---:|"]
    for name,values in comparisons.items():lines.append(f"| {name} | {values['kl']['relative']:.6%} | {values['nmse']['relative']:.6%} | {values['ppl']['relative']:.6%} | {values['top1']['absolute_pp']:.6f} |")
    lines += ["## 15. Arithmetic width implications",f"Theoretical/observed minimum signed intermediate widths: {theoretical_bits}/{observed_bits} bits.","## 16. Decision for large-batch calibration",f"**{decision}**","## 17. Limitations","This remained a small-batch pilot. Candidate Down evaluation used a bounded local shortlist after exhaustive scale-probe scoring. No dynamic runtime GPALU scale detector or floating-point hardware was modeled."]
    (OUT/"report.md").write_text("\n\n".join(lines).replace("|\n\n|","|\n|")+"\n");progress("TEST_DONE");progress("VERIFY_DONE");progress("ARTIFACT_WRITE_DONE");progress("RUN_COMPLETE")


def csv_rows(path):
    import csv
    with path.open() as f:return list(csv.DictReader(f))


if __name__=="__main__":
    try:main()
    except BaseException:progress("RUN_FAILED");raise
