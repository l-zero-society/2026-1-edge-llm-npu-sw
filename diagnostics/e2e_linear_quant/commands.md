# Commands and execution scope

From the current repository root, using the existing `.venv-calibration` environment:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/e2e-linear-pycache .venv-calibration/bin/python -u scripts/test_e2e_linear_quant.py --sequences 8 > diagnostics/e2e_linear_quant/execution_8.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/e2e-linear-pycache .venv-calibration/bin/python -u scripts/test_e2e_linear_quant.py --sequences 32 > diagnostics/e2e_linear_quant/execution_32.log 2>&1
PYTHONPYCACHEPREFIX=/tmp/e2e-linear-pycache .venv-calibration/bin/python -m unittest discover -s tests -p 'test_static_quant.py' -v > diagnostics/e2e_linear_quant/tests_static.log 2>&1
PYTHONPYCACHEPREFIX=/tmp/e2e-linear-pycache .venv-calibration/bin/python - <<'PYTHON' > diagnostics/e2e_linear_quant/tests_focused.log 2>&1
import unittest
loader=unittest.TestLoader();suite=unittest.TestSuite()
for pattern in ['test_e2e_linear_math.py','test_factorized_math.py','test_s10_search.py']:
    suite.addTests(loader.discover('tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PYTHON
```

The full-split command resumes from the first 8 per-sequence results; it does not rerun them. Both modes use every valid prefill position, without padding, token masking, reference activation injection, or KV cache. No calibration forward or parameter search is called. The source loader verifies local GGUF SHA256, numerical library versions, dataset hashes and tokenizer IDs against the original all-126 manifest. Existing calibration data may be tokenized for that loader's identity checks but never participates in an executed model forward.

A report-table whitespace fix was made after the first 8 examples; both source hashes and its non-numerical scope are recorded in `report_format_revision.json`. No quantization, model execution or metric code changed between the runs. `summary_first_8.json` preserves the initial subset results and original execution hash.

Final tests: 22 existing static_quant + 16 focused (4 new, 6 factorized, 6 s10) = 38 passing tests. An initial test filename collided with the script import; renaming the test file resolved collection before any model run.

No decode test is run: the wrapper deliberately supports full unpadded prefill only. Decode would require cache-position-aware row correction and cache-specific verification. No new large models or activation caches are generated.
