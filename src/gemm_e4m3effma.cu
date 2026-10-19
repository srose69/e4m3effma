// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E4M3EffMA GEMM host side: effma_gemm (validation, epilogue setup, routing of n <= 8 TN problems to
// the GEMV kernel) and the choice of the tiled-kernel configuration.
#include "gemm_e4m3effma.cuh"
#include "gemv_e4m3effma.cuh"

namespace effma {
namespace kern {

#define EFFMA_DECL_GEMM_T(id, ta) EFFMA_GEMM_PROTO(id, ta, 0); EFFMA_GEMM_PROTO(id, ta, 1); EFFMA_GEMM_PROTO(id, ta, 3);
#define EFFMA_DECL_GEMM(id) EFFMA_DECL_GEMM_T(id, 0) EFFMA_DECL_GEMM_T(id, 1) EFFMA_DECL_GEMM_T(id, 3)
EFFMA_GEMM_LIB(EFFMA_DECL_GEMM)
#undef EFFMA_DECL_GEMM
#undef EFFMA_DECL_GEMM_T

static cudaError_t launch_gemm(int ta, int tb, bool akc, bool bkc, const uint8_t *A, int lda, const uint8_t *B,
                               int ldb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const DeviceInfo &d = device_info();
  int id = d.table->gemm_large, bm = 128, bn = 128;
#define EFFMA_TILE(cid, cbm, cbn, cbk, ctm, ctn) if (id == cid) { bm = cbm; bn = cbn; }
  EFFMA_GEMM_CANDIDATES(EFFMA_TILE)
#undef EFFMA_TILE
  if ((long long)((m + bm - 1) / bm) * ((n + bn - 1) / bn) < 2LL * d.sms) id = d.table->gemm_small;
  using Fn = cudaError_t (*)(bool, bool, const uint8_t *, int, const uint8_t *, int, int, int, int, const Epilogue &,
                             cudaStream_t);
  const int ia = operand_index(ta), ib = operand_index(tb);
#define EFFMA_ROW(cid, t) {EFFMA_GEMM_ENTRY(cid, t, 0), EFFMA_GEMM_ENTRY(cid, t, 1), EFFMA_GEMM_ENTRY(cid, t, 3)}
#define EFFMA_CASE(cid)                                                                        \
  if (id == cid) {                                                                             \
    static const Fn f[kOperandTypes][kOperandTypes] = {EFFMA_ROW(cid, 0), EFFMA_ROW(cid, 1), EFFMA_ROW(cid, 3)}; \
    return f[ia][ib](akc, bkc, A, lda, B, ldb, m, n, k, e, s);                                 \
  }
  EFFMA_GEMM_LIB(EFFMA_CASE)
#undef EFFMA_CASE
#undef EFFMA_ROW
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace effma

extern "C" effma_status effma_gemm(effma_op transa, effma_op transb, int m, int n, int k, float alpha,
                                   const effma_tensor *A, int lda, const effma_tensor *B, int ldb, float beta,
                                   const effma_tensor *C, int ldc, const effma_tensor *D, int ldd,
                                   const effma_tensor *bias, cudaStream_t stream) {
  using namespace effma::kern;
  using effma::fail;
  effma::clear_error();
  const auto valid = [](const effma_tensor *t) { return t->dtype >= EFFMA_E4M3 && t->dtype <= EFFMA_FP32; };
  const auto operand = [](effma_dtype t) { return t == EFFMA_E4M3 || t == EFFMA_BF16 || t == EFFMA_FP32; };
  const auto aligned16 = [](const void *p) { return ((uintptr_t)p & 15u) == 0; };
  const auto op_ok = [](effma_op o) { return o == EFFMA_OP_N || o == EFFMA_OP_T; };
  if (!A || !B || !D || !A->data || !B->data || !D->data)
    return fail(EFFMA_INVALID_VALUE, "effma_gemm: A, B, D or their data is NULL");
  if (!op_ok(transa) || !op_ok(transb)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: transa / transb must be EFFMA_OP_N or EFFMA_OP_T");
  if (m < 0 || n < 0 || k < 0) return fail(EFFMA_INVALID_VALUE, "effma_gemm: negative size (m=%d n=%d k=%d)", m, n, k);
  if (!valid(D)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: invalid D dtype %d", (int)D->dtype);
  if (!operand(A->dtype) || !operand(B->dtype))
    return fail(EFFMA_NOT_SUPPORTED, "effma_gemm: A and B must be E4M3, BF16 or FP32 (got dtypes %d, %d)", (int)A->dtype,
                (int)B->dtype);
  if (m == 0 || n == 0) return EFFMA_OK;
  if (k % 16) return fail(EFFMA_INVALID_VALUE, "effma_gemm: k = %d is not a multiple of 16", k);
  if (lda % 16 || ldb % 16) return fail(EFFMA_INVALID_VALUE, "effma_gemm: lda = %d / ldb = %d must be multiples of 16", lda, ldb);
  if (!aligned16(A->data) || !aligned16(B->data)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: A and B must be 16-byte aligned");
  const bool a_kc = transa == EFFMA_OP_T, b_kc = transb == EFFMA_OP_N;
  if (lda < (a_kc ? k : m)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: lda = %d < %d", lda, a_kc ? k : m);
  if (ldb < (b_kc ? k : n)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: ldb = %d < %d", ldb, b_kc ? k : n);
  if (ldd < m) return fail(EFFMA_INVALID_VALUE, "effma_gemm: ldd = %d < m = %d", ldd, m);
  if (bias && bias->data && !valid(bias)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: invalid bias dtype %d", (int)bias->dtype);
  if (beta != 0.f && C && C->data && !valid(C)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: invalid C dtype %d", (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(EFFMA_INVALID_VALUE, "effma_gemm: ldc = %d < m = %d", ldc, m);
  const auto scaling_ok = [](const effma_tensor *t) { return t->scaling == EFFMA_SCALING_TENSOR || t->scaling == EFFMA_SCALING_OUTER; };
  if (!scaling_ok(A) || !scaling_ok(B)) return fail(EFFMA_INVALID_VALUE, "effma_gemm: invalid scaling of A or B");
  for (const effma_tensor *t : {A, B})
    if (t->scaling == EFFMA_SCALING_OUTER && (t->dtype != EFFMA_E4M3 || !t->scale_inv))
      return fail(EFFMA_INVALID_VALUE, "effma_gemm: EFFMA_SCALING_OUTER needs an E4M3 operand with a scale_inv vector");
  for (const effma_tensor *t : {C, D, bias})
    if (t && t->scaling != EFFMA_SCALING_TENSOR)
      return fail(EFFMA_INVALID_VALUE, "effma_gemm: C, D and bias take per-tensor scaling only");
  if (effma_status st = effma::check_device()) return st;
  const bool c_is_d = beta != 0.f && (!C || !C->data);  // TE "accumulate": D += ...

  Epilogue e;
  e.sa = A->dtype == EFFMA_E4M3 ? A->scale_inv : nullptr;
  e.sb = B->dtype == EFFMA_E4M3 ? B->scale_inv : nullptr;
  e.sa_vec = A->scaling == EFFMA_SCALING_OUTER;
  e.sb_vec = B->scaling == EFFMA_SCALING_OUTER;
  e.alpha = alpha; e.beta = beta;
  const effma_tensor *Cx = c_is_d ? D : C;
  e.c = beta != 0.f ? Cx->data : nullptr; e.ldc = beta != 0.f ? (c_is_d ? ldd : ldc) : 0;
  e.c_dtype = beta != 0.f ? Cx->dtype : EFFMA_FP32;
  e.c_sinv = beta != 0.f && Cx->dtype == EFFMA_E4M3 ? Cx->scale_inv : nullptr;
  e.d = D->data; e.ldd = ldd; e.d_dtype = D->dtype; e.d_scale = D->dtype == EFFMA_E4M3 ? D->scale : nullptr;
  e.amax = D->amax;
  e.bias = bias && bias->data ? bias->data : nullptr; e.bias_dtype = bias ? bias->dtype : EFFMA_FP32;

  const uint8_t *a = (const uint8_t *)A->data, *b = (const uint8_t *)B->data;
  const cudaError_t err = n <= 8 && a_kc && b_kc && A->dtype == EFFMA_E4M3
                              ? launch_gemv(B->dtype, a, lda, b, ldb, m, n, k, e, stream)
                              : launch_gemm(A->dtype, B->dtype, a_kc, b_kc, a, lda, b, ldb, m, n, k, e, stream);
  return effma::cuda_status(err, "effma_gemm: kernel launch");
}
