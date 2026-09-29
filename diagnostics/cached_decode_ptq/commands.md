# Reproduction

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/cached-ptq-pycache .venv-calibration/bin/python -u scripts/calibrate_cached_decode_ptq.py > diagnostics/cached_decode_ptq/execution.log 2>&1
```

The driver uses only the local GGUF, local OASST archive, frozen down_proj artifacts, and the native GGUF chat template. Parameter selection reads the calibration split only. Validation records are loaded after calibration inputs are fixed and cannot alter any parameter.

Focused pre-run tests:

```sh
PYTHONPYCACHEPREFIX=/tmp/cached-ptq-pycache .venv-calibration/bin/python -m unittest -v tests.test_cached_decode_ptq_math
```

Official validation was resumed from the frozen calibration decisions with all
24 recoverable chats and the established cached-decode limit of 32 response
targets per chat:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/cached-ptq-validation .venv-calibration/bin/python -u scripts/calibrate_cached_decode_ptq.py --validation-only --shard 0/1
```

An earlier mixed-mode batch acceleration attempt was rejected after it produced
a non-finite activation on the second chat. Its first record is retained as
`validation_000.batched_invalid.json` and is excluded from all official
aggregates. The official run uses the original batch-one cached-decode path.

Calibration/refinement replay and final materialization:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/cached-ptq-final .venv-calibration/bin/python -u scripts/calibrate_cached_decode_ptq.py
```

Final relevant test set:

```sh
PYTHONPYCACHEPREFIX=/tmp/cached-ptq-tests .venv-calibration/bin/python -m unittest -v tests.test_cached_decode_ptq_math tests.test_response_eval_math tests.test_row_pot_math tests.test_e2e_linear_math tests.test_factorized_math tests.test_static_quant
```

Result: 55 tests passed, 0 failed.
