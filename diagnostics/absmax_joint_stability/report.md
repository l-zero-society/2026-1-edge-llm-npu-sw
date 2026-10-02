# Joint absmax calibration stability

Calibration: 16 fixed chats, response limit 32, reservoir 64/group, seed 20261002; 45 fixed candidates/layer.
All layers satisfy selected_score <= old_score + 1e-12; no boundary expansion.

## Parameter stability

{
  "exact_sx_match_count": 6,
  "exact_s10_match_count": 8,
  "exact_pair_match_count": 2,
  "sx_direction_match_count": 7,
  "s10_direction_match_count": 11,
  "persistent_sx_boundary_count": 1,
  "persistent_s10_boundary_count": 7
}

## 384-target E2E

| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.30810588 | 201.967316 | 0 | 0 | 1 | 1 | 1 | 1 |
| old_absmax | 5.28999787 | 198.343002 | 0.0703112796 | 0.00281636199 | 0.998396437 | 0.8984375 | 1 | 0.879687507 |
| pilot_joint_absmax | 5.28713676 | 197.776332 | 0.0696620729 | 0.00283543232 | 0.998195113 | 0.8828125 | 0.997395833 | 0.886979173 |
| stability_joint_absmax | 5.38772423 | 218.705096 | 0.0672205779 | 0.00292399611 | 0.998411656 | 0.903645833 | 1 | 0.885937507 |

Old to stability: `{"kl_relative": -0.043957408969505646, "nmse_relative": 0.03821742975840897, "ppl_relative": 0.10266101504055958, "top1_pp": 0.520833333333337}`
Pilot joint to stability: `{"kl_relative": -0.03504769314887978, "nmse_relative": 0.031234667763719828, "ppl_relative": 0.10582036596866391, "top1_pp": 2.083333333333337}`

E2E gate: `{"pass": true, "kl_pass": true, "nmse_pass": true, "top1_pass": true}`
Classification: **STABILITY_PASS**.
This is a descriptive fixed-scope stability experiment on 12 conversations; it does not establish statistical significance or optimality.
