# Executed commands

Repository base commit: f31f9e6e52a65362acd473f27ed0bfc05097335f. The tree already contained the previous untracked diagnostic files. This task preserves those historical outputs and records new code hashes in provenance.json.

The previous temporary Python packages and capture files had been cleaned. An isolated environment was restored; only libraries, not model files, were downloaded. The initial sandbox package install failed DNS; the network-approved repeat succeeded.

```sh
python3 -m venv .venv-calibration
.venv-calibration/bin/python -m pip install -r scripts/requirements-quant-diagnostics.txt
.venv-calibration/bin/python -m pip freeze > diagnostics/bos_aware/python_packages.txt

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 .venv-calibration/bin/python -u scripts/cache_diagnostic_reference_inputs.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --first-layer 0 --output-dir diagnostics/bos_aware --cache-dir /tmp/bos-aware-capture --staging-dir /tmp/bos-aware-staging > diagnostics/bos_aware/capture.log 2>&1

OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/bos-aware-pycache .venv-calibration/bin/python -m unittest discover -s tests -v > diagnostics/bos_aware/tests.log 2>&1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/bos-aware-pycache .venv-calibration/bin/python -u scripts/run_bos_aware_calibration.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers all --modules down_proj > diagnostics/bos_aware/execution.log 2>&1

python3 scripts/build_bos_aware_report.py
```

Per-invocation argv/timestamps are in commands.jsonl. Capture provenance is in bulk_capture_provenance.json. Read-only inspection commands are not enumerated here.

Final verification/export and no-op resume (all exited 0):

```sh
python3 scripts/verify_bos_aware_results.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/bos-aware-pycache .venv-calibration/bin/python scripts/export_bos_aware_parameters.py > diagnostics/bos_aware/export.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/bos-aware-pycache .venv-calibration/bin/python -u scripts/run_bos_aware_calibration.py --model /Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf --layers all --modules down_proj > diagnostics/bos_aware/resume.log 2>&1
python3 scripts/build_bos_aware_report.py
git diff --check
```

Resume verification compared mtimes of all 36 frozen selection/result checkpoint files before/after the repeated command: all unchanged. See resume_verification.json. Export reloaded 18 operation policies and retained the existing unverified-profile binary export guard. Earlier export smoke testing used --output-dir diagnostics/bos_aware/export_smoke with two then-completed operations.

The numerical run covered 18 down_proj operations, four search configurations each, both complete dataset splits, and all counterfactual paths. The optional new 126-operation evaluation was not run; all 126 reference inputs are cached. Total per-operation diagnostic computation was 49.73 minutes, excluding reference capture.
