# Reproducing the local Linear diagnosis

This directory contains measurements, not a production policy. See
[quant_error_diagnosis.md](quant_error_diagnosis.md) for scope, attribution and limitations.
No model is downloaded by the diagnostic. A local GGUF with the recorded SHA256
and the original calibration/validation JSONL files are required. The original
BF16 checkpoint is not required for these local counterfactuals; its source error
is explicitly not measured.

From the repository root:

```bash
python3 -m venv /tmp/quant-diagnosis-venv
/tmp/quant-diagnosis-venv/bin/python -m pip install -r scripts/requirements-quant-diagnostics.txt

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache \
/tmp/quant-diagnosis-venv/bin/python scripts/diagnose_linear_quant_error.py \
  --model /path/to/gemma-2b-it.Q8_0.gguf \
  --layers all --modules all --integer-backend fp64-exact --blas accelerate --metric-backend native

python3 scripts/build_quant_diagnosis_report.py

PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache OPENBLAS_NUM_THREADS=2 \
/tmp/quant-diagnosis-venv/bin/python -m unittest discover -s tests -v
```

`--blas accelerate --metric-backend native` is optional and macOS-only. Else use the default `--blas numpy`.
`--integer-backend int64` (default) uses NumPy INT64 dot products and is much slower.
`fp64-exact` uses FP64 BLAS only for bounded integer operands, proves every possible
partial sum is inside INT32, checks integer-valued outputs and cross-checks
sampled full-K dots against the independent INT64 path. All counterfactual float
GEMMs remain FP64; no reduced-precision GEMM is used.

Supported targets include `--layers 8 --modules down_proj`,
`--layers all --modules down_proj`, and `--layers all --modules all`.
`--phase baseline` only measures attribution/distributions, `--phase sweep`
measures down_proj alternatives, and default `--phase all` completes all selected
baselines before sweeping. The report builder summarizes all completed results.

Resume uses an exact provenance identity (code hashes, commit, source artifact,
model/data hashes, numerical backend, block sizes and thread setting).
Each operation is checkpointed only after both splits complete and reproduce
the stored local-total MSE. An interrupted operation is recomputed. Captures
are cached in `/tmp/quant-diagnosis-capture` by layer and input group, with source
identity markers. No FP activations or model weights are added to this repository.
Full all-layer captures require approximately 67 GiB of temporary disk space.
Do not edit diagnostic or production numerical source files while a run is active.
For changed code or backend, use a new `--output-dir`; do not mix results.

Files:

- `quant_error_attribution.csv`: all selected operations and both data splits;
  MSE, MAE, NMSE, maximum absolute error and reference energy for each independent path.
- `down_proj_activation_stats.csv`: exact full-data distributions and quantization occupancy.
- `mlp_activation_comparison.csv`: gate/up/down distributions and path NMSE per layer/split.
- `down_proj_scale_sweep.csv`: calibration-fitted input thresholds, frozen s_10,
  recalculated multiplier/shift; baseline plus five exact percentiles and a
  calibration-only 64-row activation reconstruction MSE selection.
- `down_proj_output_scale_sweep.csv`: frozen baseline INT8 pair; all historical
  s_10 candidates and minmax, with clipping and requant-only/final metrics.
- `diagnosis_provenance.json`, `source_audit.json`, `commands.jsonl`: exact identities,
  historical/current implementation differences and executed diagnostic commands.
- `results/`, `sweeps/`: per-operation atomic resumable checkpoints.
- `tests.log`, `execution.log`: validation and execution records.

`actual_requant_*` is HW versus ideal INT10 in **real units**. `final_local_total_*`
is HW versus the float reference. `requantization_only_*` is HW versus the INT8
pair, whose NMSE denominator is the pair's signal energy. These are different
comparisons, never additive MSE contributions. The percentile sweep does not
refit s_10; the separate output sweep does not change s_X or s_W.

`--metric-backend native` optionally compiles `scripts/quant_metric_reduce.c` to a content-hashed temporary library using `cc -O3 -std=c11 -ffp-contract=off -shared`. It fuses FP64 metric reductions only, uses no fast-math flags, and is tested against the default NumPy reducer for every baseline/sweep comparison. No quantization, GEMM, or scale selection moves into C.

The optional `scripts/cache_diagnostic_reference_inputs.py` collects future-layer reference inputs in one prefix per sample. Its separate staging directory prevents incomplete captures from being reused, it skips publication near the currently measured layer, and records separate source/command provenance. The standalone main script does not require this optimization.

`quantization_parameters.csv` lists all 126 operations’ original scales, channel parameter ranges and full-array NPZ locations. Recreate it with `scripts/export_quant_diagnostic_parameters.py` in the diagnostic environment. `outlier_position_audit.json` records the separate BOS-position and historical output-search reproduction for layers 8 and 17.

The completed run passes `scripts/verify_quant_diagnosis.py` (coverage, scale contracts, fitting/evaluation separation, source identity and replay) and a full no-op resume check. See `output_validation.json` and `resume_validation.json`. The supplementary verification expects the parameter inventory and BOS audit to have been generated as recorded in `commands.md`.

`file_inventory.json` contains the complete list and SHA256 hashes of new repository files, including supporting scripts, reports, logs and resumable checkpoints (excluding the inventory’s own hash).
