# Absmax cross-layer E2E calibration

The final held-out evaluation is smaller than originally planned because only 10 genuinely fresh validation conversations remained after excluding all prior evaluation source indices.
It is an independent held-out exploratory validation, not a definitive model evaluation; no statistical significance or optimality is claimed.

Candidate generation used the fixed 45-pair joint grid. The 96-target selection set and 192-target held-out set are disjoint and exclude prior validation source indices.
Individually E2E-beneficial layers: [16, 17, 13, 1, 14, 0, 11]; unsafe: [2, 4, 5, 6, 7, 8, 10, 12]; non-beneficial: [3, 9, 15].
Shortlist: [16, 17, 13, 1, 14, 0]. Final configuration changes 3 layers.
Local NMSE improved while individual E2E KL failed to improve for layers: [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]. This is consistent with cross-layer interaction/error cancellation being important, but does not prove it.

| Held-out mode | NLL | PPL | KL | NMSE | Cosine | Top1 | FP top1 in top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 4.68322499 | 108.118191 | 0 | 0 | 1 | 1 | 1 | 1 |
| old_absmax | 4.68516944 | 108.328626 | 0.0681827058 | 0.00437317389 | 0.99762608 | 0.875 | 1 | 0.900000006 |
| stability_joint_absmax | 4.70110612 | 110.068855 | 0.0682230898 | 0.00461736807 | 0.99754514 | 0.885416667 | 1 | 0.888541673 |
| cross_layer_absmax | 4.65640845 | 105.257365 | 0.0638247822 | 0.00401932623 | 0.997354943 | 0.859375 | 1 | 0.896875006 |

Old to cross-layer deltas: `{"kl_relative": -0.06391538043591871, "nmse_relative": -0.08091323787531825, "ppl_relative": -0.028351335698645, "top1_pp": -1.5625}`.
Stability-joint to cross-layer deltas: `{"kl_relative": -0.06446948677189625, "nmse_relative": -0.1295200710209558, "ppl_relative": -0.04371345219517675, "top1_pp": -2.604166666666663}`.
Gate: `{"pass": false, "kl_pass": true, "nmse_pass": true, "top1_pass": false}`; classification **CROSS_LAYER_REGRESSION**, relation to old **TRADEOFF**.
Held-out metrics were computed only after selection froze and did not feed back into the search.
