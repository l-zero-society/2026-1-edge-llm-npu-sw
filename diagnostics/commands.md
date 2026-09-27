# Commands executed

Package bootstrap (isolated temporary venv; no model download):

```sh
python3 -m venv /tmp/quant-diagnosis-venv
/tmp/quant-diagnosis-venv/bin/python -m pip install --disable-pip-version-check --cache-dir /tmp/quant-diagnosis-pip-cache 'numpy==1.26.4' 'torch==2.2.2' 'transformers==4.57.6' 'gguf==0.17.1' sentencepiece protobuf
```

The initial sandbox network attempt failed DNS resolution; the same package-only install succeeded with network permission. Full resolved versions are in `python_packages.txt`.

Numerical pilot: same command as the full run below, but `--layers 8 --modules down_proj --phase baseline`, using default NumPy BLAS. Its successful measurements are archived in `pilot_numpy/`. Loader preflight failures (unsupported generic GGUF entrypoint for original Gemma) are archived in `preflight/`; the final loader uses the dedicated existing `GGUFGemmaConverter` directly.

Full measurement:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/diagnose_linear_quant_error.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers all --modules all --integer-backend fp64-exact --blas accelerate --metric-backend native > diagnostics/execution.log 2>&1
```

Tests and reporting:

```sh
PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache OPENBLAS_NUM_THREADS=2 /tmp/quant-diagnosis-venv/bin/python -m unittest discover -s tests -v > diagnostics/tests.log 2>&1
PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache python3 -m py_compile scripts/*.py
python3 scripts/build_quant_diagnosis_report.py
git diff --check
```

Diagnostic argv and invocation timestamps are also recorded in `commands.jsonl`. Read-only repository/code/environment probes are not enumerated here.

An initial Accelerate run using NumPy metric reductions completed layer 0 and was interrupted during the next capture to enable the independently validated fused reducer. Its records are archived in `pilot_accelerate_numpy/`. Final measurements are recomputed with the final recorded identity; archived pilot numbers are not merged into final CSVs.

Optional reference capture optimization, executed concurrently only for future layers:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/cache_diagnostic_reference_inputs.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --first-layer 6 > diagnostics/bulk_capture.log 2>&1
```

This publishes reference-only input caches for future layers from a single full prefix per sample. It does not replace measured results. Source hashes/argv are recorded separately in `bulk_capture_provenance.json`, and each cached input is still subject to baseline MSE reproduction checks.

Prioritized sweep after the layer-8 baseline had been independently completed:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/diagnose_linear_quant_error.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers 8 --modules down_proj --phase sweep --output-dir diagnostics/early_layer8 --integer-backend fp64-exact --blas accelerate --metric-backend native > diagnostics/early_layer8_execution.log 2>&1
```

The prioritized output is accepted only if its full provenance identity equals the main run identity. A checkpoint import audit records any such reuse; no archived pilot with a different identity is merged.

Additional completed-layer sweeps, run concurrently with remaining baselines:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/diagnose_linear_quant_error.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers 0,1,2,3 --modules down_proj --phase sweep --output-dir diagnostics/parallel_sweeps --integer-backend fp64-exact --blas accelerate --metric-backend native > diagnostics/parallel_sweeps_execution.log 2>&1
python3 scripts/import_quant_diagnostic_checkpoints.py diagnostics/parallel_sweeps
```

The importer only publishes complete, identical-identity checkpoints atomically, never overwrites completed checkpoints, and records hashes in `checkpoint_imports.json`.

After all 126 baselines completed, remaining high-layer sweeps were run separately while the main process handled lower layers:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/diagnose_linear_quant_error.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers 12,13,14,15,16,17 --modules down_proj --phase sweep --output-dir diagnostics/parallel_high_layers --integer-backend fp64-exact --blas accelerate --metric-backend native > diagnostics/parallel_high_layers_execution.log 2>&1
python3 scripts/import_quant_diagnostic_checkpoints.py diagnostics/parallel_high_layers
```

Supplementary BOS/outlier audit (reference caches and original stored parameters):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/audit_quant_diagnostic_outliers.py /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf > diagnostics/outlier_audit.log 2>&1
```

Its exact argv, script/model hashes, token verification and reproduced search scores are in `outlier_position_audit.json`.

Verified original parameter inventory:

```sh
PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python scripts/export_quant_diagnostic_parameters.py
```

Final complete-output checks and no-op resume (all commands exited successfully):

```sh
python3 scripts/verify_quant_diagnosis.py
python3 scripts/build_quant_diagnosis_report.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache /tmp/quant-diagnosis-venv/bin/python -u scripts/diagnose_linear_quant_error.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers all --modules all --integer-backend fp64-exact --blas accelerate --metric-backend native > diagnostics/resume_check.log 2>&1
PYTHONPYCACHEPREFIX=/tmp/quant-diagnosis-pycache python3 -m py_compile scripts/*.py
git diff --exit-code
git diff --check
```

Complete coverage/consistency passed: 252 attribution rows, 36 down activation rows, 108 MLP comparison rows, 252 input-sweep rows and 216 output-sweep rows. Resume exited without model loading or any changes to the 144 checkpoint files (`resume_validation.json`). All tracked production files remain unchanged. Unit-test result: 32 passed, including 7 new diagnostic tests.
