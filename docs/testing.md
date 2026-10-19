# Testing

`make test` runs everything below on GPU `$(GPU)` (default 0); the exhaustive checks take a few
seconds on a GTX 1080.

| Test | What it checks |
|---|---|
| `tests/test_gemm.cu` | `effma_gemm` bit-exact against a host emulation of the documented order: 36 configurations (4 layouts × 9 A/B dtype pairs, m/n tails, every output dtype, C of another dtype, `accumulate`, bias, scale, amax) and 2 large problems. |
| `tests/test_gemv.cu` | The GEMV path, bit-exact against an emulation of its lane order: n = 1 … 8, x in E4M3 / BF16 / FP32. |
| `tests/test_adamw.cu` | `effma_adam` bit-exact: Adam and AdamW over 12 param / grad / moment dtype combinations with the E4M3 copy and amax. |
| `tests/test_e4m3effma.cu` | Exhaustive over 2³² inputs each: one MAC (decode → FFMA → RNE) against an exact integer reference; FP32 → BF16 rounding and FP32 → E4M3 encoding on every FP32 bit pattern; `effma_cast` on 1 M random bit patterns. |
| `test_e4m3effma -DEFFMA_NEGATIVE_CONTROLS` | Four deliberately broken variants (truncation instead of RNE, flush-to-zero, RNE without NaN handling, E4M3 ties away from zero) must be rejected. |
| `tests/check_ref.py` | The integer reference model itself against Python `Fraction` (1.38 M cases). |
| `binds/python/tests` | The PyTorch bindings (`pytest`): results against a float64 reference, `torch._scaled_mm`, and rejection of bad input (dtype, device, shape, scales, overlapping output). |

The reference model (`tests/e4m3effma_ref.h`) holds the exact sum `ACC + A·B` as a 320-bit integer in
units of 2⁻¹³³ and rounds it with integer arithmetic only, so it shares no floating-point code with the
implementation it checks.
