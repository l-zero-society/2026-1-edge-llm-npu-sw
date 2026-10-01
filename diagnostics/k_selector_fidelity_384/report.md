# Fixed 384-target selector confirmation

| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.30810588 | 201.967316 | 0 | 0 | 1 | 1 | 1 | 1 |
| Final MSE3 | 5.35498076 | 211.659904 | 0.066937372 | 0.00318137056 | 0.997662802 | 0.890625 | 0.997395833 | 0.881250007 |
| HW absmax | 5.28999787 | 198.343002 | 0.0703112796 | 0.00281636199 | 0.998396437 | 0.8984375 | 1 | 0.879687507 |

| MSE3 → absmax | Delta |
|---|---:|
| kl_relative % | 5.040395 |
| nmse_relative % | -11.473312 |
| ppl_relative % | -6.291651 |
| top1_pp pp | 0.781250 |

| Predefined criterion | Result |
|---|---|
| KL degradation <=10% | PASS |
| NMSE degradation <=10% | PASS |
| Top1 degradation <=1pp | PASS |
| Overall | HW-ready |

Classification: {'case': 'A', 'label': 'HW-ready'}. No follow-up calibration was run.
NLL outcomes: {'better': 9, 'worse': 3, 'equal': 0}. KL outcomes: {'absmax_lower': 5, 'absmax_higher': 7, 'approximately_equal': 0}.
Direct absmax vs MSE3: {'nmse': 0.0020134736007674935, 'cosine': 0.9983925289086093, 'top1': 0.90625}.

| Paired bootstrap difference | Mean | 95% low | 95% high |
|---|---:|---:|---:|
| kl | 0.00337390763 | -0.00509115076 | 0.0129297294 |
| nmse | -0.000280112109 | -0.00100950559 | 0.000305842631 |
| top1 | 0.0078125 | -0.0182291667 | 0.0338541667 |

Descriptive equal-conversation mean differences. NMSE CI uses per-conversation NMSE, not pooled-energy NMSE. Top1 units are fractions.
Seed 20261001; 10,000 paired resamples. CI does not change the decision rule.
The first 8 chats reproduce prior FP/MSE3/absmax NLL; maximum per-chat difference 0. Their fresh KL relative delta is 11.472260%.
All 3 paths freshly executed with independent caches, layers=False throughout; 12 chats × 32 = 384 response targets.
Frozen fingerprints unchanged. Previous 256-target outputs, including their original not-yet verdict, were not modified.
No parameter selection, histogram, hidden capture, long validation or generation.
