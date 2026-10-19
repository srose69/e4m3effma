# C API

Everything is declared in [`include/e4m3effma.h`](../include/e4m3effma.h), which also documents the
exact order of operations. The semantics follow cuBLASLt / Transformer Engine (`nvte_cublas_gemm_v2`):
column-major matrices, per-tensor or row-wise (outer) scales.

```c
typedef struct {
  void *data;              /* device pointer */
  effma_dtype dtype;       /* EFFMA_E4M3, EFFMA_BF16, EFFMA_FP16, EFFMA_FP32 */
  const float *scale_inv;  /* E4M3 input:  dequantization factor (device, 1 float), NULL = 1 */
  const float *scale;      /* E4M3 output: quantization factor   (device, 1 float), NULL = 1 */
  float *amax;             /* optional: max |value| before quantization, max-reduced into it */
  effma_scaling scaling;   /* EFFMA_SCALING_TENSOR (0, default) or EFFMA_SCALING_OUTER */
} effma_tensor;
```

With `scaling = EFFMA_SCALING_OUTER` an E4M3 GEMM operand's `scale_inv` is a vector along its outer
dimension: `m` floats for A (one per row of D), `n` floats for B (one per column of D); the factor of
`D[i, j]` is then `(α·sa[i])·sb[j]`, applied once to the FP32 sum (scales along K cannot be). C, D, the
bias and `effma_cast` / `effma_adam` operands take per-tensor scales only. Initializers that leave the
field out get per-tensor scaling.

| Function | Description |
|---|---|
| `effma_gemm` | `D = α·(sa·sb)·op(A)·op(B) + bias + β·C` (`sa`, `sb` per tensor or per row / column), column-major. A, B ∈ {E4M3, BF16, FP32} in any combination and any N/T layout; D ∈ {E4M3, BF16, FP16, FP32}. C may have any dtype, may alias D, and `C = NULL` with β ≠ 0 accumulates into D (TE `accumulate`). For `n ≤ 8`, TN layout and an E4M3 A the bandwidth-bound GEMV kernel runs. |
| `effma_cast` | Element-wise conversion between E4M3, BF16, FP16 and FP32 with `scale_inv` / `scale` / `amax`. FP32 → E4M3 is round-to-nearest-even with saturation (`cvt.rn.satfinite.e4m3` semantics). |
| `effma_adam` | In-place Adam / AdamW step. Params FP32 or BF16, grads FP32 / BF16 / FP16, moments FP32 or BF16, optional E4M3 copy of the updated weights with `scale` and `amax`. |
| `effma_status_string` | Name of a status code. |
| `effma_last_error` | Description of the last failure on the calling thread: the invalid argument, the CUDA error, or a device the build does not cover (sm_89+ with native FP8, or an architecture missing from `ARCHS`). |

**Constraints of `effma_gemm`** (cuBLASLt-like): A and B 16-byte aligned; `k`, `lda` and `ldb`
multiples of 16; `m`, `n`, `ldc`, `ldd` arbitrary (`ldc, ldd ≥ m`); no E4M3 NaN codes in the inputs.
"TN" (`transa = T`, `transb = N`) is the FP8 layout of cuBLASLt / TE: both operands K-contiguous.

## Errors

Every function returns an `effma_status` (`EFFMA_OK`, `EFFMA_INVALID_VALUE`, `EFFMA_NOT_SUPPORTED`,
`EFFMA_CUDA_ERROR`) and validates its arguments before launching anything. On failure
`effma_last_error()` says what was wrong:

```
effma_gemm: k = 24 is not a multiple of 16
effma_cast: invalid dtype (src 3, dst 7)
device 0 is sm_61, but this build of the library has no code for it: rebuild with ARCHS including 61
```

## Example

```c
#include "e4m3effma.h"

// Linear layer forward, TN layout as in TE: W is [out, in] E4M3, X is [in, batch] BF16, Y = W·X in BF16.
effma_tensor W = {d_w, EFFMA_E4M3, d_w_scale_inv, NULL, NULL};   // data, dtype, scale_inv, scale, amax
effma_tensor X = {d_x, EFFMA_BF16, NULL, NULL, NULL};
effma_tensor Y = {d_y, EFFMA_BF16, NULL, NULL, d_y_amax};
effma_status st = effma_gemm(EFFMA_OP_T, EFFMA_OP_N, out, batch, in, 1.0f, &W, in, &X, in,
                             0.0f, NULL, 0, &Y, out, /*bias=*/NULL, stream);
if (st != EFFMA_OK) fprintf(stderr, "%s: %s\n", effma_status_string(st), effma_last_error());

// Weight gradient accumulated into an FP32 main_grad (TE "accumulate"): G += dY·Xᵀ
effma_tensor dY = {d_dy, EFFMA_BF16, NULL, NULL, NULL};          // [out, batch]
effma_tensor G  = {d_main_grad, EFFMA_FP32, NULL, NULL, NULL};   // [out, in]
effma_gemm(EFFMA_OP_N, EFFMA_OP_T, out, in, batch, 1.0f, &dY, out, &X, in,
           1.0f, /*C = D*/ NULL, 0, &G, out, NULL, stream);

// AdamW step on BF16 weights + FP32 moments, refreshing the E4M3 copy used by the next forward pass
effma_tensor P = {d_param, EFFMA_BF16, NULL, NULL, NULL};
effma_tensor M = {d_exp_avg, EFFMA_FP32, NULL, NULL, NULL}, V = {d_exp_avg_sq, EFFMA_FP32, NULL, NULL, NULL};
effma_adam_config cfg = {.lr = 1e-3f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f,
                         .weight_decay = 0.01f, .step = t, .adamw = 1, .bias_correction = 1, .grad_scale = 1.0f};
effma_tensor Wq = {d_w, EFFMA_E4M3, NULL, d_w_scale, d_w_amax};  // E4M3 copy: quantization scale + amax
effma_adam(&cfg, numel, &P, &G, &M, &V, &Wq, stream);
```

The library is a plain C ABI, so it can also be called from any language with a C FFI.
