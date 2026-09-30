# Commands executed

```bash
.venv-calibration/bin/python scripts/finalize_downproj_ptq.py --phase step1
.venv-calibration/bin/python scripts/finalize_downproj_ptq.py --phase short
.venv-calibration/bin/python scripts/finalize_downproj_ptq.py --phase long
.venv-calibration/bin/python scripts/finalize_downproj_ptq.py --phase generation
.venv-calibration/bin/python scripts/finalize_downproj_ptq.py --phase aggregate
PYTHONPYCACHEPREFIX=/tmp/final-downproj-tests .venv-calibration/bin/python -m unittest tests.test_finalize_downproj_ptq tests.test_cached_decode_ptq_math tests.test_response_eval_math tests.test_row_pot_math tests.test_e2e_linear_math tests.test_factorized_math tests.test_static_quant
```

The long and generation stages used their per-conversation JSON checkpoints under `progress/`; aggregation did not rerun inference.
