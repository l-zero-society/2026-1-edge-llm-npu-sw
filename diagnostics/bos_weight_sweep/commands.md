# Reproduction

Repository HEAD: `f31f9e6e52a65362acd473f27ed0bfc05097335f` (existing working-tree changes retained).

From repository root:

```sh
python3 scripts/analyze_bos_weight_sweep.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/bos-weight-pycache .venv-calibration/bin/python -u scripts/analyze_bos_weight_sweep.py --evaluate-missing > diagnostics/bos_weight_sweep/validation_execution.log 2>&1
```

The first command only rescores stored calibration metrics and freezes all policies. The second reuses that immutable selection and evaluates missing validation metrics only. Existing validation is reused on subsequent runs. No large cached data is copied. The optional validation path requires the original local GGUF, dataset and activation-cache paths from the previous experiment.

After adding the answer appendix, report-only regeneration (no GEMMs):

```sh
python3 - <<'PYTHON'
import sys
sys.path.insert(0, 'scripts')
from analyze_bos_weight_sweep import load_analysis, render
render(*load_analysis())
PYTHON
```

Final whitespace verification: `git diff --check`.
