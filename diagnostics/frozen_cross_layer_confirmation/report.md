# Frozen cross-layer absmax confirmation

This was confirmation only. Layers 13, 16, and 17 were frozen before dataset inspection; no parameter was tuned on confirmation data.
Dataset: `local_unseen`; 32 sequences and 1024 aligned targets. All local examples were excluded by stable identity if previously used.

| Mode | NLL | PPL | KL | Logits NMSE | MSE | MAE | Cosine | Flattened cosine | Top1 | FP top1 in Q top5 | Top5 overlap | FP energy | Quant energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.361759 | 213.099458 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 342.454227 | 342.454227 |
| old_absmax | 5.39421604 | 220.129506 | 0.0630182347 | 0.00346830927 | 1.18773717 | 0.747463738 | 0.998806315 | 0.998372028 | 0.884765625 | 0.994140625 | 0.883593757 | 342.454227 | 331.386995 |
| frozen_cross_layer_absmax | 5.39694191 | 220.730369 | 0.0639165307 | 0.00299244547 | 1.0247756 | 0.708269301 | 0.998867919 | 0.998611413 | 0.891601562 | 0.997070312 | 0.883007819 | 342.454227 | 331.498275 |

Old to frozen-cross deltas: `{"kl_relative": 0.01425454094484346, "nmse_relative": -0.13720339270535079, "ppl_relative": 0.0027295899236697493, "top1_pp": 0.68359375}`.
Safety gate: `{"pass": true, "kl_pass": true, "nmse_pass": true, "top1_pass": true}` — **CONFIRMATION_PASS**.
Strict comparison: **TRADEOFF**.
Bootstrap (sequence unit, 2000 replicates): `{"seed": 20261004, "replicates": 2000, "unit": "sequence", "intervals": {"kl_relative": {"mean": 0.014281516100790296, "low": -0.047143948705952905, "high": 0.08102109251992565}, "nmse_relative": {"mean": -0.13613240272817537, "low": -0.18947050683416097, "high": -0.07567036973185254}, "top1_pp": {"mean": 0.663720703125, "low": -0.87890625, "high": 2.1484375}, "ppl_relative": {"mean": 0.0024897259888118526, "low": -0.024927020633195768, "high": 0.03182825408661981}}, "favorable_fraction": {"kl": 0.3325, "nmse": 1.0, "top1": 0.8285}}`.
Cross-layer KL was lower for 16/32 sequences.
No statistical significance or optimality is claimed; parameters were not changed after evaluation.
