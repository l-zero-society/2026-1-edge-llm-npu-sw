# Joint absmax calibration pilot

Calibration-only joint output-domain selection; fixed small subsets and ranges.

| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | In top5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.774858979282081 | 322.0990088820862 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| old_absmax | 5.690623345605019 | 296.0781221265844 | 0.06476878252118029 | 0.0027491388958027696 | 0.9991778645911512 | 0.8402777777777778 | 1.0 | 0.8805555624680387 |
| joint_calibrated_absmax | 5.743712755980912 | 312.2214638320372 | 0.06789535284391938 | 0.002975926666779412 | 0.9975826209818693 | 0.8611111111111112 | 0.9930555555555556 | 0.8833333398732874 |
| previous sequential (stored): calibrated_absmax | 5.7732249177121675 | 321.573109063326 | 0.06558285767779638 | 0.0031744532588783153 | 0.9984500326482295 | 0.875 | 0.9861111111111112 | 0.8638888967947828 |

{
  "changed_sx_layers": 10,
  "changed_s10_layers": 17,
  "sx_boundary_hits": 1,
  "s10_boundary_hits": 9,
  "boundary_hit_layers": 9,
  "mean_relative_local_nmse_change": -0.07374938220400908,
  "median_relative_local_nmse_change": -0.06551587436731532,
  "best_layer_improvement": 0.1974149215853424,
  "worst_layer_change": -0.0004375943425313012,
  "total_feasible_candidates": 810,
  "total_rejected_candidates": 0,
  "total_unique_acc": 72
}

{
  "kl_relative_percent": 4.827279749648934,
  "nmse_relative_percent": 8.249411163724362,
  "ppl_relative_percent": 5.452392628507333,
  "top1_pp": 2.083333333333337
}

JOINT_REGRESSION: REGRESSION
All 18 layers satisfy the local no-regression runtime invariant. E2E does not select parameters.
Previous sequential metrics are stored reference only. No automatic expansion or further calibration.
