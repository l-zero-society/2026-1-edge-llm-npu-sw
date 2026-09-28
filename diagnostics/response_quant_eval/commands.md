# Reproduction

From the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 PYTHONPYCACHEPREFIX=/tmp/response-eval-pycache .venv-calibration/bin/python -u scripts/evaluate_response_quant.py > diagnostics/response_quant_eval/execution.log 2>&1
```

The driver loads only the local recorded GGUF, extracts its native `tokenizer.chat_template`, and calls the tokenizer's `apply_chat_template`. The existing validation anchors and locally available ready-tree archive supply actual conversation history and responses. Missing anchors are skipped. For an existing user anchor, choose the eligible human assistant child by rank, then message ID; no model score or quantization parameter is selected.

All ancestor turns are preserved; only the final assistant content is a target. No sequence truncation is applied. Token offsets and exact tokenized-prefix equality verify the first response content token and exclude prompt, template, EOS/turn markers and padding. Masks and token IDs are in examples.json. Existing calibration and validation conversation trees are disjoint.

The eight focused mask/math tests run before any evaluation forward. For each example Stage 1/2 run FP, BOS-only and General in that order. Keeping one example in RAM and sharing FP lm_head chunks avoids both duplicate reference forwards and large logits/activation dumps. Every mode sees identical ground-truth history and target positions.

Stage 3 is explicitly bounded to the first 16 valid examples and first 32 response targets each. Full teacher-forcing metrics on exactly this window are retained for cache parity. This is a deployment-path check, not a second full-response PPL estimate. Change --decode-examples / --decode-tokens only for a separately identified run. Each mode uses its own KV cache and absolute positions. Original prompt caches are copied in RAM and reused for Stage 4 rather than recomputing prompts. Stage 4 uses the same 16 prompts, greedy argmax, max_new_tokens=32 and the existing model generation_config EOS IDs. EOS comparisons are censored when EOS is not observed.

Final report regeneration (no model loading):

```sh
PYTHONPYCACHEPREFIX=/tmp/response-eval-pycache .venv-calibration/bin/python scripts/evaluate_response_quant.py --report-only
```

No production defaults, quantization parameters, calibration artifacts, prior diagnostics, RTL, LUT, QB or ABI files are modified. Raw full-sequence PPL is not reported as the official evaluation metric.

Final relevant tests (after Stages 3/4; 52 passed, zero failures):

```sh
PYTHONPYCACHEPREFIX=/tmp/response-eval-pycache .venv-calibration/bin/python - <<'PY' > diagnostics/response_quant_eval/tests.log 2>&1
import unittest
patterns=['test_response_eval_math.py','test_row_pot_math.py','test_e2e_linear_math.py','test_factorized_math.py','test_static_quant.py','test_s10_search.py']
suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p) for p in patterns)
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not r.wasSuccessful())
PY
```

Post-run analysis reads only the saved per-example statistics/CSV files: generation means/medians, cache/full parity extrema, count/mask assertions, and 61 source/artifact/data hash checks. Results are saved as analysis.json and verification.json. answers.md supplies the report interpretation; --report-only appends it without executing model inference. analysis.json and verification.json are also included in the final summary JSON after report regeneration.
