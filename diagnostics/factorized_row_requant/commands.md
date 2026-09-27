# Factorized row experiment commands

No installation, download, activation capture or model forward was run. Existing local caches, model weights and environment were reused.

```sh
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/factorized-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_factorized_math.py -v > diagnostics/factorized_row_requant/tests.log 2>&1
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/factorized-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_static_quant.py -v >> diagnostics/factorized_row_requant/tests.log 2>&1
OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/factorized-pycache .venv-calibration/bin/python -m unittest discover -s tests -p test_bos_calibration.py -v >> diagnostics/factorized_row_requant/tests.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/factorized-pycache .venv-calibration/bin/python -u scripts/test_factorized_row_requant.py --layers 7,8,17 > diagnostics/factorized_row_requant/execution_3layer.log 2>&1
# Only after reviewing the positive three-layer results and recording decision.json:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/factorized-pycache .venv-calibration/bin/python -u scripts/test_factorized_row_requant.py --layers all > diagnostics/factorized_row_requant/execution_all_down.log 2>&1
```

The optional common-s10 optimization was not run. The supplemental `output_grid_bound.json` is an analytical lower bound, not a fourth architecture scheme: for cached validation BOS rows in layers 7/8/17, recover the same GGUF W, compute `Y_ref = dot(X_BOS, W)`, then compare `clip(rint(Y_ref / frozen_s10), -512, 511) * frozen_s10` against Y_ref using ErrorMetric. It involves no parameter selection or scale change.
