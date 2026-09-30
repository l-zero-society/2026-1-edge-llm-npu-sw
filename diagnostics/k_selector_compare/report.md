# Frozen k-selector comparison

| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | FP-top1-in-Q-top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.0066132 | 149.3979 | 0 | 0 | 1 | 1 | 1 | 1 |
| Final MSE3 | 5.0289087 | 152.76621 | 0.067078059 | 0.0037504944 | 0.99746679 | 0.88541667 | 0.99739583 | 0.88880209 |
| HW absmax | 4.9780898 | 145.19676 | not measured | not measured | not measured | not measured | not measured | not measured |



Stored short JSONs contain aggregate metrics only, no aligned FP logits or hidden vectors. New HW-vs-FP KL/NMSE/cosine/top-k and hidden NMSE cannot be recovered without forbidden FP inference. These metrics are unmeasured, not zero.



PPL relative change: -4.95492%.

Conversation NLL outcomes: {'better': 16, 'worse': 7, 'equal': 1}.

Selector rates include all prompt prefill rows and 31 cached decode rows per chat; both selectors see the identical HW trajectory activation.

All-layer selector statistics: {'layer': 'all', 'rows': 187182, 'agreement': 0.9417946170037718, 'mean_abs_delta': 0.058205382996228267, 'max_abs_delta': 1, 'clamp_rate': 0.00012821745680674423, 'hw_clip_rate': 5.291265653214518e-05, 'mse3_clip_rate': 4.684507424592108e-05, 'hw_zero_rate': 0.41396085299730845, 'mse3_zero_rate': 0.4147615185432619, 'effective_shift_min': 10, 'effective_shift_max': 29}

Delta k histogram (HW minus MSE3): {'-1': 6496, '0': 176287, '1': 4399}

Clipping/zero-rate differences compare both selectors on identical HW-trajectory inputs, not the independently propagated MSE3 trajectory.

| Layer | k agreement | Mean absolute delta k | HW clipping | MSE3 clipping | HW zero | MSE3 zero | Shift min | Shift max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8853736 | 0.1146264 | 4.245873e-05 | 3.016835e-05 | 0.7526612 | 0.7625136 | 14 | 28 |
| 1 | 0.9296086 | 0.07039138 | 6.496174e-05 | 4.474777e-05 | 0.5540451 | 0.5620264 | 13 | 29 |
| 2 | 0.9494182 | 0.05058179 | 5.578798e-05 | 4.731266e-05 | 0.4737861 | 0.4772916 | 14 | 27 |
| 3 | 0.9468218 | 0.05317819 | 4.46069e-05 | 4.451299e-05 | 0.3882499 | 0.3829975 | 16 | 28 |
| 4 | 0.9468218 | 0.05317819 | 4.793481e-05 | 4.672573e-05 | 0.3125575 | 0.3077909 | 14 | 27 |
| 5 | 0.9428791 | 0.05712088 | 4.912629e-05 | 4.752983e-05 | 0.2919817 | 0.2876851 | 10 | 27 |
| 6 | 0.9488412 | 0.05115877 | 5.101034e-05 | 5.014755e-05 | 0.301765 | 0.2977768 | 14 | 26 |
| 7 | 0.9387441 | 0.06125589 | 5.557081e-05 | 5.117468e-05 | 0.2664924 | 0.2639948 | 16 | 26 |
| 8 | 0.9448024 | 0.05519762 | 6.104689e-05 | 5.669772e-05 | 0.2280802 | 0.225826 | 13 | 27 |
| 9 | 0.9432638 | 0.05673622 | 6.017823e-05 | 5.384523e-05 | 0.239387 | 0.2380354 | 11 | 26 |
| 10 | 0.9425906 | 0.05740937 | 6.108798e-05 | 5.493692e-05 | 0.2487665 | 0.2470172 | 15 | 25 |
| 11 | 0.9423021 | 0.05769786 | 5.501909e-05 | 5.315852e-05 | 0.3120062 | 0.3090457 | 14 | 27 |
| 12 | 0.9472065 | 0.05279354 | 4.907933e-05 | 4.529361e-05 | 0.3725144 | 0.3698547 | 14 | 27 |
| 13 | 0.954611 | 0.04538898 | 4.725984e-05 | 4.487689e-05 | 0.4501101 | 0.4489705 | 15 | 27 |
| 14 | 0.9462448 | 0.05375517 | 4.45306e-05 | 3.818586e-05 | 0.6092338 | 0.6158718 | 12 | 26 |
| 15 | 0.9387441 | 0.06125589 | 5.598753e-05 | 4.468907e-05 | 0.5733726 | 0.5822071 | 15 | 26 |
| 16 | 0.9535532 | 0.04644677 | 5.2912e-05 | 4.531122e-05 | 0.5363481 | 0.541136 | 15 | 27 |
| 17 | 0.950476 | 0.04952399 | 5.386871e-05 | 4.389671e-05 | 0.5399377 | 0.5456661 | 13 | 29 |
| all | 0.9417946 | 0.05820538 | 5.291266e-05 | 4.684507e-05 | 0.4139609 | 0.4147615 | 10 | 29 |

Hidden comparison omitted: aligned FP hidden vectors were not saved in the baseline artifacts.

Absmax is a promising HW candidate on this 768-target likelihood test: PPL changes by -4.9549% and k agrees on 94.1795% of identical-input rows. This does not establish FP-logit fidelity or hardware adoption; the required aligned FP reference vectors are absent and FP inference was explicitly forbidden.

No recalibration, FP/MSE3 inference, long validation or generation. Only 18 down_proj patched. Frozen parameter and weight fingerprints verified before/after every conversation.
