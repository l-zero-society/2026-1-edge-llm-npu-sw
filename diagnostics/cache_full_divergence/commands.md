# Exact execution

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/cache-full-pycache .venv-calibration/bin/python -u scripts/diagnose_cache_full_divergence.py > diagnostics/cache_full_divergence/execution.log 2>&1
```

The run reuses stored chat token IDs/masks from response_quant_eval/examples.json (first four valid prompts, 16 response targets each). It checks the native GGUF template and frozen calibration-only w_BOS=0.25 selection. Model/artifact/source identity is recorded in provenance.json. No prior diagnostic files are written.

Full-path references use the exact prompt plus t previous GT response tokens, for t=0..15. Target 0 is predicted by prefill, followed by 15 one-token cache updates. Full reference vectors needed by interventions stay only in RAM and are reused across all controls. Only compact scalar traces, code mismatch statistics and sufficient statistics are written to disk. No full activations, logits, ACC or KV tensors are saved. The same prefix at t=0 makes all three interventions identities; cached prefill and its untouched KV are reused.

Run order: FP, BOS-only, General-row, force-full-k, force-full-Xq+k, force-full-q10. Each control has its own evolving cached history; only the current matched down_proj predictor row is intervened on. No full-path residual, hidden state or KV is injected. Per-token K/V statistics are collected during General-row baseline, without replaying it. Full and cached calls use the existing FP transformer and unchanged quantization helpers. Full calls temporarily enable cache production to observe post-RoPE K/V; that cache is discarded and never used by the cached path.

A syntax error in the first launch was corrected before any model execution. The successful run's script hash is in provenance.json.

Focused and relevant tests, after the trace/control run:

```sh
PYTHONPYCACHEPREFIX=/tmp/cache-full-pycache .venv-calibration/bin/python - <<'PY' > diagnostics/cache_full_divergence/tests.log 2>&1
import unittest
patterns=['test_cache_full_trace_math.py','test_response_eval_math.py','test_row_pot_math.py','test_e2e_linear_math.py','test_factorized_math.py','test_static_quant.py']
suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p) for p in patterns)
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not r.wasSuccessful())
PY
```

Report generation after tests, without inference:

```sh
PYTHONPYCACHEPREFIX=/tmp/cache-full-pycache .venv-calibration/bin/python scripts/diagnose_cache_full_divergence.py --report-only
```

Control recovery:

The three complete baseline traces were saved before an indexing error in the new control wrapper (`x[-1:0]` was empty). It was corrected to select the last row explicitly. Eight pure control/math tests passed before resuming. This changes no baseline arithmetic. baseline_provenance.json and baseline_execution.log preserve the original completed baseline run. Since reference vectors were intentionally RAM-only, termination required rebuilding the 64 General FULL references; FP/BOS baselines and all completed baseline statistics were reused. Each control still starts from its own freshly computed cached prefill, never from a FULL KV cache.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/cache-full-pycache .venv-calibration/bin/python -u scripts/diagnose_cache_full_divergence.py --resume-controls > diagnostics/cache_full_divergence/execution.log 2>&1
```
