// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMM instance object: -DEFFMA_ID=<config> -DEFFMA_TA=<A dtype> -DEFFMA_TB=<B dtype>, all 4
// layouts.  Small objects keep the build parallel; the kernel bodies are compiled only for the
// architectures whose table (or the defaults) select the configuration, trapping stubs elsewhere.
#include "gemm_e4m3effma.cuh"

namespace effma {
namespace kern {

#define EFFMA_CFG_(id) GemmCfg##id
#define EFFMA_CFG(id) EFFMA_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemm : EFFMA_CFG(EFFMA_ID) {
  template <int TA, int TB> struct On { static constexpr bool value = gemm_selected(EFFMA_ARCH_CC, ID); };
};
using C = LibGemm<EFFMA_ID>;

EFFMA_GEMM_PROTO(EFFMA_ID, EFFMA_TA, EFFMA_TB) {
  if (akc) return bkc ? gemm_launch<C, EFFMA_TA, EFFMA_TB, true, true>(A, lda, B, ldb, m, n, k, e, s)
                      : gemm_launch<C, EFFMA_TA, EFFMA_TB, true, false>(A, lda, B, ldb, m, n, k, e, s);
  return bkc ? gemm_launch<C, EFFMA_TA, EFFMA_TB, false, true>(A, lda, B, ldb, m, n, k, e, s)
             : gemm_launch<C, EFFMA_TA, EFFMA_TB, false, false>(A, lda, B, ldb, m, n, k, e, s);
}

}  // namespace kern
}  // namespace effma
