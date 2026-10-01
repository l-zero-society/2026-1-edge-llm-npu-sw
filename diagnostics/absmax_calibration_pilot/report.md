# Absmax-aware calibration pilot

Calibration: changed sX 12/18, changed s10 17/18; boundary-hit layers 10/18.
Mean local NMSE change: -7.045313% relative.

| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.77485898 | 322.099009 | 0 | 0 | 1 | 1 | 1 | 1 |
| old_absmax | 5.69062335 | 296.078122 | 0.0647687825 | 0.0027491389 | 0.999177865 | 0.840277778 | 1 | 0.880555562 |
| calibrated_absmax | 5.77322492 | 321.573109 | 0.0655828577 | 0.00317445326 | 0.998450033 | 0.875 | 0.986111111 | 0.863888897 |

Deltas old→calibrated: {'kl_relative': 0.012568943323118335, 'nmse_relative': 0.15470821198772158, 'top1_pp': 3.472222222222221, 'ppl_relative': 0.08610898621493404}.
Pilot classification: REGRESSION.
This is a fixed small pilot; no statistical claim or automatic large calibration follows.
Frozen source artifacts and parameters remained unchanged.
