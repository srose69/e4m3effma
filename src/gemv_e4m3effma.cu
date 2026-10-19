// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E4M3EffMA GEMV host side: the configuration the device's table selects per x dtype and n.
#include "gemv_e4m3effma.cuh"

namespace effma {
namespace kern {

#define EFFMA_DECL_GEMV_T(id, t)                                                                   \
  EFFMA_GEMV_PROTO(id, t, 1); EFFMA_GEMV_PROTO(id, t, 2); EFFMA_GEMV_PROTO(id, t, 3); EFFMA_GEMV_PROTO(id, t, 4); \
  EFFMA_GEMV_PROTO(id, t, 5); EFFMA_GEMV_PROTO(id, t, 6); EFFMA_GEMV_PROTO(id, t, 7); EFFMA_GEMV_PROTO(id, t, 8);
#define EFFMA_DECL_GEMV(id) EFFMA_DECL_GEMV_T(id, 0) EFFMA_DECL_GEMV_T(id, 1) EFFMA_DECL_GEMV_T(id, 3)
EFFMA_GEMV_LIB(EFFMA_DECL_GEMV)
#undef EFFMA_DECL_GEMV
#undef EFFMA_DECL_GEMV_T

cudaError_t launch_gemv(int xt, const uint8_t *A, int lda, const uint8_t *X, int ldx, int m, int n, int k,
                        const Epilogue &e, cudaStream_t s) {
  const int id = device_info().table->gemv[operand_index(xt)][n - 1];
  using Fn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, int, int, const Epilogue &, cudaStream_t);
#define EFFMA_ROW(cid, t)                                                                      \
  {EFFMA_GEMV_ENTRY(cid, t, 1), EFFMA_GEMV_ENTRY(cid, t, 2), EFFMA_GEMV_ENTRY(cid, t, 3), EFFMA_GEMV_ENTRY(cid, t, 4), \
   EFFMA_GEMV_ENTRY(cid, t, 5), EFFMA_GEMV_ENTRY(cid, t, 6), EFFMA_GEMV_ENTRY(cid, t, 7), EFFMA_GEMV_ENTRY(cid, t, 8)}
#define EFFMA_CASE(cid)                                                                        \
  if (id == cid) {                                                                             \
    static const Fn f[kOperandTypes][8] = {EFFMA_ROW(cid, 0), EFFMA_ROW(cid, 1), EFFMA_ROW(cid, 3)}; \
    return f[operand_index(xt)][n - 1](A, lda, X, ldx, m, k, e, s);                            \
  }
  EFFMA_GEMV_LIB(EFFMA_CASE)
#undef EFFMA_CASE
#undef EFFMA_ROW
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace effma
