// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// effma_gemm on the GEMV path (n <= 8, TN, E4M3 A), bit-exact against the host emulation of the
// kernel's lane order: n = 1..8, x in {E4M3, BF16, FP32}, every output dtype, accumulate, bias.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const effma_dtype outs[] = {EFFMA_BF16, EFFMA_FP32, EFFMA_FP16, EFFMA_E4M3};
  const effma_dtype ops3[] = {EFFMA_E4M3, EFFMA_BF16, EFFMA_FP32};
  const int gs[][2] = {{1, 16}, {33, 272}, {300, 1024}, {1000, 4112}};
  for (effma_dtype bt : ops3)
    for (int n = 1; n <= 8; ++n) {
      const int *g = gs[(n + bt) % 4];
      Case c{EFFMA_OP_T, EFFMA_OP_N, g[0], n, g[1], EFFMA_E4M3, bt, outs[idx % 4], n == 3 ? -1 : (int)outs[idx % 4],
             idx % 3 == 0 ? -1 : (int)EFFMA_BF16, idx % 2 ? 0.75f : 1.f, n == 3 ? 1.f : (idx % 4 == 1 ? -1.5f : 0.f)};
      fails += run_case(c);
      if (n == 1 || n == 4 || n == 8) { c.vec = 1 + (n & 2) + (bt == EFFMA_E4M3 ? 2 : 0); fails += run_case(c); }
      ++idx;
    }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
