# BOS-aware offline static scale calibration

The runtime contract is unchanged: symmetric INT8 [-127,127] inputs and per-output-channel INT8 weights, INT32 accumulation, one static INT10 [-512,511] output scale, UInt16 multiplier, UInt5 shift and zero point zero. No RTL, QB layout or LUT contents change. Measurements remain local Linear operations on reference-model prefill inputs.

Enable the new activation search in the existing real-model CLI:

```sh
# Add to the existing src/calibrate_gemma.py command:
--activation-search output-mse --scale-mode mse \
--search-rows 64 --bos-search-rows 16 --early-search-rows 16 \
--early-position-limit 8 --activation-percentiles 99,99.5,99.9,99.95,99.99
```

`search_rows` remains 64 by default. Activation search is explicit (`none` remains the default). Representative row sampling and the final-output MSE objective apply even when activation search is disabled. `--uniform-search-sampling` explicitly disables stratification for comparisons; `--bos-search-rows 0` disables the BOS guarantee. Consumer constraints take priority: Q/K use the intended fixed s_10=0.2, and gate uses the existing GeLU LUT scale s_10=0.1. Only s_X is searched for these operations. Other operations follow the global output-scale mode. The old Q/K bug discarded the per-operation options; it is fixed.

## Representation without over-weighting BOS

`RepresentativeRowSampler` keeps independent bounded priority reservoirs for valid position 0 (the structural BOS proxy), positions 1..early_position_limit and general positions. The seed determines selection reproducibly, independently of input batching. The requested quotas are filled when possible, and unavailable capacity is redistributed to other groups. The total selected rows never exceeds the budget. At least one row from each nonempty sampling stratum is required; a budget too small for that fails explicitly. BOS is guaranteed whenever observed and enabled. Disabled strata join the general pool.

Each selected row receives objective weight `N_group / (N_total * n_selected_group)`. These weights sum to one and restore the true calibration population proportions. Selecting 16 BOS rows out of 64 does not assign BOS 25% of the output-error objective. BOS, early/general population counts, sampled counts, row IDs, position histogram and weights are recorded. The Gemma hook propagates positions while excluding padding. Legacy callbacks without position metadata are marked unknown and sampled uniformly; they do not claim a BOS guarantee.

Position affects calibration representation only. Runtime uses one static s_X and s_10 per operation, with no BOS state or per-token scale.

## Candidate search

The input candidates are absmax and the configured percentiles. For every candidate, the existing quantize_input, quantize_weight, exact accumulator and hardware profile apply() compute the final INT10 output. Candidate selection minimizes population-weighted `mean((Y_hw - X_float @ W_float.T)^2)`, not activation reconstruction MSE or requantization-only MSE.

The bounded output grid uses the original absmax-input path's full-calibration dequantized-accumulator absmax/511, multiplied by `{1,.95,.9,.8,.7,.5}`. The common grid is reused across input candidates. Fixed consumer scales reduce this to one output candidate. Unrepresentable multiplier/shift candidates are excluded; if none are valid, calibration fails. Ties retain deterministic candidate order. All-zero scales use the existing sentinel 1; a zero percentile on nonzero data falls back to absmax and records the requested/effective thresholds.

The generic streaming CLI estimates percentiles from the existing scalar reservoir (`--percentile-capacity`, default 65536). At p99.99 this contains only about 6.6 upper-tail observations, so estimator uncertainty must be considered. The real diagnostic runner uses **exact full-calibration percentiles** over disk-cached inputs. These are different candidate-estimation modes, clearly recorded; the same sampler and joint-search implementation select the pair. A larger streaming scalar reservoir should be validated separately before claiming the exact diagnostic's selected scales will reproduce through the generic CLI.

Same-input operations share Pass A statistics, not necessarily their selected s_X. A quantized input cache cannot be reused across operations with different s_X. Existing per-operation scale metadata carries this distinction.

## Memory and artifacts

The sampler retains at most `3 * search_rows * K` activation elements. For FP32 down_proj K=16384, 64/128/256 search rows require at most 12/24/48 MiB across the three reservoirs; the final selected rows occupy 4/8/16 MiB. Priority/position/ID storage and bounded merge temporaries are additional. Search is chunked across output channels; its M dimension is bounded by the selected row count. Production evaluation retains existing M/N/K chunking.

The real experiment additionally uses approximately 67 GiB of temporary disk-backed reference captures for all Linear input groups, and one ~2.2 GiB temporary for an exact down_proj percentile calculation. These diagnostic caches are not runtime memory or model artifacts.

`scale_search.json` contains complete candidate scores and sampling metadata. `manifest.json` and `report.json` summarize the selected pair and link to that table. Full calibration and validation position-group metrics are separate from sample-based candidate estimates. Validation never participates in parameter selection. The final output report includes BOS, non-BOS, early, general and total MSE/NMSE, with null for absent/undefined groups rather than NaN/Inf.

The original 126-operation files are preserved. They recorded Q/K MSE scales, not the intended fixed 0.2; their dirty historical GGUF entrypoint is not completely available in the checkout. The new real runner verifies the recorded local GGUF/data/token hashes and reproduces historical MSE before reporting improvements. Its optional Accelerate FP64 backend computes bounded integer dots exactly and checks sampled full-K dots against NumPy INT64. The generic production implementation remains NumPy INT64.

## Measured policy and promotion limits

The [real OASST1 report](../diagnostics/bos_aware_calibration_report.md) covers all 18 down_proj operations with 64/128/256 search rows and a uniform control. Exact executed commands are in [commands.md](../diagnostics/bos_aware/commands.md). The runner supports `--layers 8 --modules down_proj`, `--layers all --modules down_proj`, and `--layers all --modules all`; completed operations resume only when source/model/data identities match. No-op resume was checked without rewriting the frozen selections or results.

The unchanged 64-row default is the main reported/exported policy. A 256-row quality preset is recommended for further qualification: it finds a materially better layer-14 threshold that 64/128 miss, with bounded additional memory and measured search cost. This is an explicit recommendation, not a silent default change. The streaming percentile estimator must still be qualified against the exact offline experiment.

Guaranteed BOS coverage does not guarantee BOS fidelity under total MSE. Layer 7/8 BOS errors worsen despite total improvements; layer 17 retains poor ordinary-token accuracy. Do not automatically promote clipping as a production default. Define calibration-only acceptance criteria for important token groups and test additional thresholds between p99.99 and absmax before promotion. There is no group-specific runtime scale or new hardware state.
