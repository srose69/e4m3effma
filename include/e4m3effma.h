// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* E4M3EffMA: FP8 (E4M3) GEMM / GEMV / cast / Adam for NVIDIA GPUs without native FP8
 * (sm_50 ... sm_87).
 *
 * GEMM semantics follow cuBLASLt / Transformer Engine (nvte_cublas_gemm_v2), column-major:
 *
 *     D = alpha * (sa * sb) * op(A) * op(B) + bias + beta * C          D, C: m x n
 *
 *   op(A) is m x k: transa = EFFMA_OP_T -> A stored k x m (lda >= k, K contiguous)
 *                   transa = EFFMA_OP_N -> A stored m x k (lda >= m)
 *   op(B) is k x n: transb = EFFMA_OP_N -> B stored k x n (ldb >= k, K contiguous)
 *                   transb = EFFMA_OP_T -> B stored n x k (ldb >= n)
 *   A, B: E4M3, BF16 or FP32 in any combination (e.g. E4M3 weights x BF16/FP32 activations).
 *   sa, sb = scale_inv of E4M3 operands (1 if NULL or not E4M3): one factor per tensor, or with
 *   scaling = EFFMA_SCALING_OUTER one per row of op(A) (m floats) / per column of op(B) (n floats),
 *   as the row-wise scaling of cuBLASLt / torch._scaled_mm.  bias has length m.
 *   "TN" (transa = T, transb = N) is the FP8 layout of cuBLASLt / TE: both operands K-contiguous.
 *
 * Accumulation (TE `accumulate`): C may alias D, and C == NULL with beta != 0 means C = D, so
 * beta = 1 adds the product into D in place -- e.g. FP32 main_grad += dY^T X.  C may have any
 * dtype (E4M3 C is dequantized with C->scale_inv).
 *
 * Arithmetic: finite E4M3 x E4M3 products are exact in FP32, E4M3 / BF16 products whenever they are
 * finite in FP32 (FP32 x FP32 products round as in SGEMM); accumulation is one FP32 FFMA per MAC in a fixed order (ascending k for the tiled kernel;
 * k = 256c + 8l + j per lane then an xor-butterfly for the GEMV kernel).  Epilogue, in FP32:
 *     v = acc * s  (s = (alpha * sa[i]) * sb[j]);  v = v + bias[i];  v = beta * C + v (one FMA);
 *     amax = max(amax, |v|);  D = convert(v)   (BF16 / FP16 / FP32: RNE; E4M3: RNE satfinite of
 *                                               v * D->scale, NaN -> 0x7F)
 *
 * Dispatch: n <= 8, transa = T, transb = N and A E4M3 run the bandwidth-bound GEMV kernel;
 * everything else runs the tiled GEMM kernel.  Kernel configurations are chosen per compute
 * capability (tuned with `make tune`, defaults otherwise).
 *
 * Requirements (cuBLASLt-like FP8 constraints): A and B 16-byte aligned; k, lda, ldb multiples of 16.
 * m, n, ldc, ldd are unrestricted (ldc, ldd >= m).  Operands must not contain the E4M3 NaN code.
 */
#ifndef E4M3EFFMA_H
#define E4M3EFFMA_H

#define E4M3EFFMA_VERSION "0.1.2"

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { EFFMA_E4M3 = 0, EFFMA_BF16 = 1, EFFMA_FP16 = 2, EFFMA_FP32 = 3 } effma_dtype;
typedef enum { EFFMA_OP_N = 0, EFFMA_OP_T = 1 } effma_op;
typedef enum { EFFMA_OK = 0, EFFMA_INVALID_VALUE = 1, EFFMA_NOT_SUPPORTED = 2, EFFMA_CUDA_ERROR = 3 } effma_status;
/* How scale_inv of an E4M3 GEMM operand is applied: one factor for the whole tensor, or one per
 * outer index (row of op(A), column of op(B)); never along K. */
typedef enum { EFFMA_SCALING_TENSOR = 0, EFFMA_SCALING_OUTER = 1 } effma_scaling;

typedef struct {
  void *data;              /* device pointer */
  effma_dtype dtype;
  const float *scale_inv;  /* E4M3 input:  dequantization factor (device, 1 float), NULL = 1 */
  const float *scale;      /* E4M3 output: quantization factor   (device, 1 float), NULL = 1 */
  float *amax;             /* optional output statistic: max |v| before quantization (device,
                              1 float, max-reduced into the existing value); NULL = off */
  effma_scaling scaling;   /* E4M3 GEMM operands: EFFMA_SCALING_OUTER makes scale_inv a vector
                              (m floats for A, n for B); everything else: EFFMA_SCALING_TENSOR (0) */
} effma_tensor;

/* D = alpha * (sa * sb) * op(A) * op(B) + bias + beta * C; see the header comment. */
effma_status effma_gemm(effma_op transa, effma_op transb, int m, int n, int k, float alpha,
                        const effma_tensor *A, int lda, const effma_tensor *B, int ldb, float beta,
                        const effma_tensor *C, int ldc, const effma_tensor *D, int ldd,
                        const effma_tensor *bias, cudaStream_t stream);

/* Element-wise conversion of count values: v = src * src.scale_inv (E4M3 source) ;
 * dst = convert(v) (E4M3 destination: RNE satfinite of v * dst.scale); dst.amax gets max |v|. */
effma_status effma_cast(const effma_tensor *src, const effma_tensor *dst, size_t count, cudaStream_t stream);

/* Adam / AdamW step over count elements, in place (PyTorch / TE fused-Adam semantics).
 *   param       FP32 or BF16 (updated); grad FP32, BF16 or FP16 (e.g. the BF16 / FP32 output of
 *               effma_gemm); exp_avg, exp_avg_sq FP32 or BF16 (updated)
 *   param_e4m3  optional: E4M3 copy of the updated FP32 value for the next FP8 pass, quantized
 *               with param_e4m3->scale; param_e4m3->amax gets max |param|
 * Per element, in FP32 (bit-reproducible; host emulation in tests/test_adamw.cu):
 *   g = grad * grad_scale;  if (!adamw) g = weight_decay * p + g (one FMA)
 *   m = beta1 * m + (1 - beta1) * g;       v = beta2 * v + (1 - beta2) * (g * g)   (one FMA each)
 *   u = (m * rbc1) / (sqrt(v * rbc2) + eps)  with rbc = 1 / (1 - beta^step) (1 if no bias corr.)
 *   if (adamw) p = p - (lr * weight_decay) * p (one FMA);   p = p - lr * u (one FMA)
 * The scalar factors (1 - beta, rbc, lr * weight_decay) are formed on the host in double and
 * rounded to FP32 once. */
typedef struct {
  float lr, beta1, beta2, eps, weight_decay;
  int step;             /* 1-based step number */
  int adamw;            /* 1: decoupled weight decay (AdamW); 0: L2 term added to the gradient */
  int bias_correction;  /* 1: Adam bias correction */
  float grad_scale;     /* gradient multiplier, e.g. 1 / loss_scale (0 is treated as 1) */
} effma_adam_config;

effma_status effma_adam(const effma_adam_config *cfg, size_t count, const effma_tensor *param,
                        const effma_tensor *grad, const effma_tensor *exp_avg,
                        const effma_tensor *exp_avg_sq, const effma_tensor *param_e4m3,
                        cudaStream_t stream);

const char *effma_status_string(effma_status status);

/* Description of the most recent failure of an effma_* call on the calling thread (which argument
 * was wrong, or the CUDA error); "" after a successful call.  Valid until the next effma_* call. */
const char *effma_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* E4M3EFFMA_H */
