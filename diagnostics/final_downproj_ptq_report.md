# Final down_proj PTQ validation

Boundary layers: [0, 1, 3, 7, 8, 10, 17].

## Boundary decisions

| layer | boundary i | outward i | relative gain % | accepted | final s10 |
|---:|---:|---:|---:|:---:|---:|
| 0 | 8 | 9 | -0.0742 | no | 0.0558736429 |
| 1 | 8 | 13 | 8.0833 | yes | 0.0240541647 |
| 3 | 8 | 12 | 2.2909 | yes | 0.0116903393 |
| 7 | -8 | -16 | 4.9426 | yes | 0.00438090005 |
| 8 | -8 | -16 | 13.1038 | yes | 0.00691621142 |
| 10 | -8 | -9 | -0.1176 | no | 0.00256533268 |
| 17 | -8 | -16 | 36.0731 | yes | 0.104500894 |

## Final parameters

| layer | sX base | final s10 |
|---:|---:|---:|
| 0 | 0.27601293 | 0.0558736429 |
| 1 | 0.103824543 | 0.0240541647 |
| 2 | 0.0628007074 | 0.0094859994 |
| 3 | 0.042651045 | 0.0116903393 |
| 4 | 0.0382233959 | 0.00610538903 |
| 5 | 0.0282725307 | 0.00418232583 |
| 6 | 0.027339089 | 0.00376797484 |
| 7 | 0.0334963346 | 0.00438090005 |
| 8 | 0.0206241448 | 0.00691621142 |
| 9 | 0.0315673476 | 0.00297051675 |
| 10 | 0.0345791929 | 0.00256533268 |
| 11 | 0.0346520991 | 0.0070843014 |
| 12 | 0.0477467785 | 0.0078883462 |
| 13 | 0.053734252 | 0.00924775857 |
| 14 | 0.134304021 | 0.00644720279 |
| 15 | 0.124244062 | 0.0137368282 |
| 16 | 0.117047629 | 0.0299505427 |
| 17 | 0.101945936 | 0.104500894 |

## Short cached validation

| mode | targets | NLL | PPL | KL | logits NMSE % | cosine | top1 % | in5 % | overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 768 | 5.006613 | 149.397904 | 0.000000 | 0.000000 | 1.00000000 | 100.000 | 100.000 | 100.000 |
| old_general | 768 | 5.022560 | 151.799450 | 0.074223 | 0.470313 | 0.99724546 | 88.932 | 99.740 | 88.464 |
| calibrated | 768 | 5.028909 | 152.766208 | 0.067078 | 0.375049 | 0.99746679 | 88.542 | 99.740 | 88.880 |

## Long cached validation

| mode | targets | NLL | PPL | KL | logits NMSE % | cosine | top1 % | in5 % | overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 14630 | 4.008573 | 55.068254 | 0.000000 | 0.000000 | 1.00000000 | 100.000 | 100.000 | 100.000 |
| previous | 14630 | 4.073682 | 58.772976 | 0.064109 | 0.699359 | 0.99797983 | 88.640 | 99.563 | 88.319 |
| final | 14630 | 4.074411 | 58.815819 | 0.059625 | 0.633777 | 0.99819377 | 88.934 | 99.604 | 88.846 |

## Response depth

| bucket | targets | FP NLL | previous NLL | final NLL | previous KL | final KL | previous NMSE % | final NMSE % | previous top1 % | final top1 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-31 | 768 | 5.006613 | 5.022560 | 5.028909 | 0.074223 | 0.067078 | 0.470313 | 0.375049 | 88.932 | 88.542 |
| 32-63 | 763 | 4.821992 | 4.933956 | 4.924489 | 0.069206 | 0.062803 | 0.535143 | 0.454059 | 88.729 | 89.253 |
| 64-127 | 1299 | 5.301393 | 5.392265 | 5.397013 | 0.077816 | 0.072642 | 0.531103 | 0.437440 | 86.143 | 87.067 |
| 128-255 | 2292 | 4.329629 | 4.403688 | 4.406446 | 0.066690 | 0.061615 | 0.466501 | 0.405043 | 88.351 | 88.133 |
| 256-511 | 3877 | 3.862441 | 3.931912 | 3.936059 | 0.066486 | 0.062542 | 1.114338 | 1.019634 | 88.342 | 88.470 |
| >=512 | 5631 | 3.433932 | 3.486807 | 3.484043 | 0.056190 | 0.052356 | 0.523107 | 0.489738 | 89.487 | 90.020 |

## Greedy-64

| mode | mean prefix | median prefix | agreement % | exact32 | exact64 | mean abs length delta |
|---|---:|---:|---:|---:|---:|---:|
| FP | 59.7500 | 64.0000 | 100.000 | 16/16 | 13/16 | 0.000 |
| previous | 17.6250 | 6.0000 | 32.116 | 4/16 | 1/16 | 6.562 |
| final | 21.6250 | 16.5000 | 50.166 | 4/16 | 2/16 | 2.938 |

## Answers

A. Better outward candidates were found for layers **[1, 3, 7, 8, 17]**.

B. The >=1% rule accepted layers **[1, 3, 7, 8, 17]**; all other boundary layers retained their incumbent s10.

C. All final s10 values are listed in the Final parameters table and CSV artifact.

D. Short validation improved KL (0.074223 -> 0.067078) and logits NMSE (0.470313% -> 0.375049%), while PPL rose slightly (151.799450 -> 152.766208).

E. Long validation evaluated **14630** identical response targets.

F. FINAL PPL is **58.815819**, versus FP **55.068254** (absolute gap 3.747564).

G. Versus previous, FINAL KL changed 0.064109 -> 0.059625, logits NMSE 0.699359% -> 0.633777%, and top1 88.640% -> 88.934%.

H. Error does not grow monotonically with response depth; FINAL improves KL and logits NMSE in every depth bucket, including >=512.

I. At sampled response positions >=128 the worst FINAL hidden-state layer is **layer 16** at **1.718611% NMSE**.

J. Greedy-64 improves overall: mean shared prefix 17.625 -> 21.625, median 6.000 -> 16.500, and agreement 32.116% -> 50.166%.

K. FINAL is stable enough to freeze as the completed down_proj software baseline: KL/NMSE/top1 and greedy preservation improve, while the long PPL change is negligible. This is not an RTL adoption decision.

Conversation NLL outcomes: 11 better, 11 worse, 2 approximately equal.
