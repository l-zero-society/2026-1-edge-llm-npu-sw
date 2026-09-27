# Reproduction

From the repository root with the existing local environment:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/row-pot-pycache .venv-calibration/bin/python -u scripts/test_e2e_row_pot.py --sequences 32 > diagnostics/e2e_row_pot/execution.log 2>&1
PYTHONPYCACHEPREFIX=/tmp/row-pot-pycache .venv-calibration/bin/python - <<'PYTHON' > diagnostics/e2e_row_pot/tests.log 2>&1
import unittest
suite=unittest.TestSuite();loader=unittest.TestLoader()
for pattern in ['test_row_pot_math.py','test_e2e_linear_math.py','test_factorized_math.py','test_static_quant.py']:
    suite.addTests(loader.discover('tests',pattern=pattern))
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not r.wasSuccessful())
PYTHON
```

The numerical script is resumable by completed sequence. It reruns FP plus BOS-only and compares every stored per-sequence hidden/logit sufficient statistic with the previous E2E records before accepting the new general-row result. It never reruns all-Linear mode. Only down_proj artifacts are loaded and checked; all other operations execute in the original FP model.

No calibration activations are captured, no calibration search is run, and no parameters are selected from reference Linear/logit outputs. The only new choice is the requested activation-only k selection on the current propagated activation row. Base scales, common s10, weight arrays and channel factors remain frozen. Artifact/source hashes and the repository commit are in provenance.json.

Optional arbitrary-row oracle is not implemented or measured; the experiment remains focused on the specified three-candidate PoT rule. Zero activation rows define k0=0; ties prefer k0, then k0-1, then k0+1. A missing feasible candidate raises an explicit error instead of silently expanding the candidate search or clamping k.

Final report-only regeneration, without model execution:

```sh
PYTHONPYCACHEPREFIX=/tmp/row-pot-pycache .venv-calibration/bin/python - <<'PYTHON'
import sys
from pathlib import Path
sys.path.insert(0,'scripts')
from test_e2e_row_pot import materialize
materialize(Path('diagnostics/e2e_row_pot'))
PYTHON
```
