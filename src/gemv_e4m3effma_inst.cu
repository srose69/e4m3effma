// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMV instance object: -DEFFMA_ID=<config> -DEFFMA_TX=<x dtype>, n = 1..8.  The kernel bodies are
// compiled only for the architectures whose table (or the defaults) select the configuration for that
// x dtype and n, trapping stubs elsewhere.
#include "gemv_e4m3effma.cuh"

namespace effma {
namespace kern {

#define EFFMA_CFG_(id) GemvCfg##id
#define EFFMA_CFG(id) EFFMA_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemv : EFFMA_CFG(EFFMA_ID) {
  template <int NB, int XT> struct On { static constexpr bool value = gemv_selected(EFFMA_ARCH_CC, ID, XT, NB); };
};
using C = LibGemv<EFFMA_ID>;

#define EFFMA_GEMV_DEF(nb) \
  EFFMA_GEMV_PROTO(EFFMA_ID, EFFMA_TX, nb) { return gemv_launch<C, nb, EFFMA_TX>(A, lda, X, ldx, m, k, e, s); }
EFFMA_GEMV_DEF(1) EFFMA_GEMV_DEF(2) EFFMA_GEMV_DEF(3) EFFMA_GEMV_DEF(4)
EFFMA_GEMV_DEF(5) EFFMA_GEMV_DEF(6) EFFMA_GEMV_DEF(7) EFFMA_GEMV_DEF(8)

}  // namespace kern
}  // namespace effma
