# Executed commands

No activation capture, model forward, model download, LUT generation or binary export was performed. The local GGUF is read only to recover the same weights/token positions. Main-pass s_X/k and calibration row IDs/weights are copied from the prior frozen experiment. The three-layer expansion decision was recorded before the all-down and joint runs.

```sh
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_s10_search.py -v > diagnostics/s10_calibration/tests.log 2>&1
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_factorized_math.py -v >> diagnostics/s10_calibration/tests.log 2>&1
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_static_quant.py -v >> diagnostics/s10_calibration/tests.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -u scripts/test_s10_calibration.py --layers 7,8,17 > diagnostics/s10_calibration/execution_3layer.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -u scripts/test_s10_calibration.py --layers all > diagnostics/s10_calibration/execution_all_down.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/s10-pycache .venv-calibration/bin/python -u scripts/test_s10_calibration.py --layers 7,8,17 --joint > diagnostics/s10_calibration/execution_joint.log 2>&1
git diff --check
```

Every numerical command exited 0. Main results include only frozen-s10 and s10-only policies. The optional joint results are separately identified and were not mixed into the all-18 summary. Detailed assertions and counts are recorded in verification.json. The report generator materialize() was rerun after writing answers.md; it performs no numerical recalibration.
