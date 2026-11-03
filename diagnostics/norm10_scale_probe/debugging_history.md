# Norm10 scale probe debugging history

## Original recorded failure

The requested scale-probe directory initially contained only `pid` and an empty
`run.log`; it had no status, FAILED, capture or numeric traceback. PID 98026 was
not alive. Therefore no scale-probe-specific original overflow can be asserted.

The preserved historical `diagnostics/norm10_debug_pilot/run.log`, lines 10–21,
does record `OverflowError: math range error` at
`scripts/evaluate_response_quant.py:223`, function `merged`, from `math.exp(nll)`.
Its callers were `run_norm10_debug_pilot.py:178` (`measure`) and line 238
(`sweeps`), in CUMULATIVE_SWEEP. This is derived-PPL float64 range exhaustion,
not evidence of accumulator overflow. Exact original NLL/logits were not saved
at the failing point; they cannot be recovered from the traceback. The code
uses torch float64 log-softmax and Python float aggregation; the diagnostic had
two target predictors. The actual failing logits shape and numeric NLL are
unknown, rather than guessed from the architecture.

A deterministic minimal score record with shape (1,8) float64 operands and
NLL=1000 reproduces the original aggregation OverflowError. Float64 exp has
maximum finite log input approximately 709.782712893384. This reproducer is
independent of model loading and hardware arithmetic.

## Debug iteration 1: derived metric and integer-boundary fixes

Regression tests were written first. The original response.merged demonstrably
raises on the finite NLL=1000 record. The new merge_scores adapter reuses the
existing accumulators and preserves NLL as log_ppl; PPL is null with an explicit
`exceeds_float64_range` label when mathematically outside float64. No NLL or
logit is clipped, and a nonfinite primary metric still fails explicitly.
The historical scoring implementation is unchanged.

Audit also found that the inherited input quantizer casts to np.int64 before
INT10 saturation: finite input +1e30 exceeds the intermediate's representation.
The exact cast behavior is platform dependent; on this Mac it clips at INT64_MAX
and the final INT10 sign happens to remain correct. The regression demonstrates
the out-of-range intermediate, not an observed wrong-sign model activation.
The probe's new input
quantizer rounds and saturates before integer conversion. Nonfinite activations
are rejected explicitly; finite quotient overflow saturates with its exact sign.
OLD_EMULATOR remains unmodified as requested. Static output gains validate
positive finite s_out before division. Signed arbitrary-precision integer
products, RNE, RTL shifts and all hardware widths are retained.

## Execution and scope

The runner's os.setsid call was removed. The foreground command directly owns
the interpreter and is awaited. An identified old launchd job,
gui/501/com.lzero.norm10-debug-pilot (then PID 33980, command
run_norm10_debug_pilot.py), was booted out to stop its automatic restarts and
concurrent historical-log mutation. No unrelated process was terminated.

Existing staged/unstaged changes and historical artifacts are preserved.
Publication uses a detached Git worktree only after scientific DONE. The worktree
is a Git checkout, not a detached experiment process.

## Remaining limitations

The original failing NLL and logits were not logged. Historical combined Norm
modes also changed Softmax. This probe keeps Softmax and RoPE FP, uses 24 aligned
targets, and does not establish production RTL integration or broad accuracy.
Null PPL is a representational limitation with finite log_ppl, not fabricated
finite precision. Hardware findings, if any, remain in accumulator_verification.

Observed hardware overflow: False.
Final small E2E completed targets: 24; error: None.
