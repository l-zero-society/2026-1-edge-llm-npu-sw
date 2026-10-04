# Activation LUT pilot blocked during bounded discovery

**Historical block resolved:** the subsequent explicit user contract specifies a shared signed INT10 input scale, `q + 512` addressing, and a shared signed INT8 output scale. FP16 expansion is not required for the resumed pilot. The original discovery evidence below is retained for provenance; current status is in `discovery.json`.

The required previous **FP16-expansion LUT implementation and address semantics could not be located**. The available integer LUT generator does not establish that path, and its hardware format is explicitly unconfirmed. Proceeding would require inventing an input conversion or substituting a different LUT contract.

Confirmed from the recorded model configuration and installed Gemma implementation:

```text
gate = gate_proj(x)
up = up_proj(x)
act = GELUTanh(gate)  # torch.nn.functional.gelu(..., approximate="tanh")
h = act * up
y = down_proj(h)
```

The activation consumes FP `gate_proj` output. Frozen `down_proj.sx` describes the downstream product `h`; `down_proj.s10` describes `y`. Neither establishes the activation LUT input scale.

The available `src/LUT.py` generator and `tests/test_static_quant.py::test_existing_lut_contract` establish a different software table: 1,024 signed-code addresses, `x = signed_code * 0.1`, tanh GeLU, and unsigned 8-bit entries `clip(round(y * 255), 0, 255)`. This is integer decoding, not FP16 expansion. The generator explicitly marks the proposed 10-bit-input/8-bit-output hardware interpretation as uncertain and requiring confirmation. The roadmap mentions FP16 expansion but supplies no implementation; its INT10-output examples also do not resolve the intended output contract.

To unblock this specified pilot, the authoritative FP16 input/address/expansion implementation (including domain behavior) and intended activation table output format are needed. No replacement addressing, width, or input scale was assumed.

Discovery evidence and frozen-file SHA256 values are in `discovery.json`. Work stopped before model loading, capture, calibration, inference, or LUT generation. The branch is `lut_cali`; its initial working tree was clean, and `diagnostics/**/work/` was already ignored. `frozen_linear_quant_v1` and existing artifacts were not modified. No experiment driver or numerical tests were created against an unknown contract.
