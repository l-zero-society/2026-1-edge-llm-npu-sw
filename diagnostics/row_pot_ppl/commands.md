# Reproduction

Main sequential run, repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/row-ppl-pycache .venv-calibration/bin/python -u scripts/diagnose_row_pot_ppl.py > diagnostics/row_pot_ppl/execution.log 2>&1
```

The default order is baseline verification/A, B, strict C caps, D2/D3, calibration-only E followed by E validation, then F. Individual stages can resume with `--stage ab|c|d|e|f`; completed per-sequence records are reused only with identical provenance. `--stage report` reads results without loading the model. FP forwards are shared across modes within a stage. Existing full FP logits/hidden states were not persisted by earlier experiments, so new mode comparisons require a fresh FP forward; no large new reference cache is created.

C uses the literal k0±1 intersection with the cap and hardware shift range. An empty intersection makes that policy undefined; it is logged without adding a fallback candidate, silently clamping k0, or extending the search. Other experiments continue. D2/D3 are row-local FP-weight output oracles and never receive labels or reference-trajectory activations. F uses arbitrary per-row absmax and an ideal FP64 real ratio, not packed hardware M/S.

E samples existing calibration caches only: all 128 BOS rows plus 256 uniformly sampled non-BOS rows (seed 42), with population weights. All nine requested quarter-octave base scales are evaluated per operation; the lowest estimated pooled calibration final-output NMSE is frozen before E validation. It is not an exhaustive full-token calibration sweep. Input-cache identities, selected row IDs/weights/data hashes, candidates and decisions are in recenter_selection.json.

Token CSV uses predictor p -> target p+1 and associates k with the predictor. No future target-row k is used. JSON/NPZ checkpoints contain only compact metrics, selected k, confusion counts and target-logit/LSE/NLL values; no captured activations or vocabulary-sized logits are stored.


Tests executed after F completed:

```sh
PYTHONPYCACHEPREFIX=/tmp/row-ppl-pycache .venv-calibration/bin/python - <<'PYTEST' > diagnostics/row_pot_ppl/tests.log 2>&1
import unittest,sys
names=['test_row_pot_ppl_math.py','test_row_pot_math.py','test_e2e_linear_math.py','test_factorized_math.py','test_static_quant.py','test_s10_search.py']
suite=unittest.TestSuite()
for name in names:suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
PYTEST
```

Result: 52 passed, 0 failed (8 new tests). Runtime invariants and source/artifact hash checks are in verification.json.

Report regeneration executed without model loading:

```sh
PYTHONPYCACHEPREFIX=/tmp/row-ppl-pycache .venv-calibration/bin/python scripts/diagnose_row_pot_ppl.py --stage report
```

Final analysis postprocessing uses only saved metrics: position_nll.json groups token CSV by predictor_position==0; general_minus_bos.json joins the two baseline rows on (sequence,predictor_position) and subtracts their NLL. answers.md contains the report interpretation. The delivered summary incorporates these derived records and verification.json; failed-cap CSV/report rows contain blank/N/A numerical values with explicit status. Re-running --stage report regenerates raw aggregates plus answers.md, with failure reasons in the summary and prose; these presentation-only enrichments do not change any measured metric or selected scale.
