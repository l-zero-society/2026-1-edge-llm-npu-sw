# Fixed-subset selector fidelity confirmation
| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.57809414 | 264.566897 | 0 | 0 | 1 | 1 | 1 | 1 |
| Final MSE3 | 5.60479341 | 271.725784 | 0.0648239373 | 0.00343108216 | 0.997565231 | 0.87890625 | 1 | 0.880468757 |
| HW absmax | 5.54477489 | 255.896968 | 0.0722607083 | 0.00265951664 | 0.999059104 | 0.90234375 | 1 | 0.875000007 |

| MSE3 → absmax | Delta |
|---|---:|
| KL relative % | 11.472260 |
| NMSE relative % | -22.487527 |
| PPL relative % | -5.825291 |
| Top1 pp | 2.343750 |

| Decision criteria | Result |
|---|---|
| KL <= +10% | FAIL |
| NMSE <= +10% | PASS |
| abs(Top1 delta) <= 1 pp | FAIL |
| Overall | not-yet |

Subset [0, 3, 6, 9, 12, 15, 18, 21]: 8 conversations, 256 response targets. Each mode ran fresh with independent KV caches.
Direct absmax-vs-MSE3 logits comparison: {'nmse': 0.002189253912676573, 'cosine': 0.9979616520420957, 'top1': 0.90234375}.
Conversation NLL outcomes: {'better': 6, 'worse': 2, 'equal': 0}.
Largest hidden NMSE degradation: layer 3, MSE3 1.254374%, absmax 1.399203% (delta 0.144829 pp). Hidden positions only 0, 7, 31.
Fresh/stored FP and MSE3 NLL: all absolute differences <1e-6; maximum 0.
Frozen decisions, weights, sX/s10 and channel M/S fingerprints: unchanged. No recalibration, search, long validation or greedy generation.
The HW-ready label applies only to the requested fixed-subset fidelity heuristic; PPL was excluded from the decision.
