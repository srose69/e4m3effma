// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// effma_gemm, tiled kernel, bit-exact against the host emulation: all 4 layouts x A, B in
// {E4M3, BF16, FP32}, m/n tails, ld padding, every output dtype, C of another dtype, TE-style
// accumulate (C = NULL, D += ...), alpha / beta / bias / scale / amax, and two larger problems.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const effma_dtype outs[] = {EFFMA_BF16, EFFMA_FP32, EFFMA_FP16, EFFMA_E4M3};
  const effma_dtype ops3[] = {EFFMA_E4M3, EFFMA_BF16, EFFMA_FP32};
  const int sizes[][3] = {{1, 9, 16}, {37, 45, 48}, {200, 300, 144}, {257, 129, 256}};
  // tiled GEMM: layouts x operand types x tails; outputs, C dtype, accumulate, bias rotate
  for (effma_op ta : {EFFMA_OP_N, EFFMA_OP_T})
    for (effma_op tb : {EFFMA_OP_N, EFFMA_OP_T})
      for (effma_dtype at : ops3)
        for (effma_dtype bt : ops3) {
          const int *sz = sizes[idx % 4];
          Case c{ta, tb, sz[0], sz[1] < 9 ? 9 : sz[1], sz[2], at, bt, outs[idx % 4],
                 idx % 5 == 1 ? -1 : (idx % 5 == 2 ? (int)outs[(idx + 1) % 4] : (int)outs[idx % 4]),
                 idx % 3 == 0 ? -1 : (idx % 3 == 1 ? (int)EFFMA_BF16 : (int)EFFMA_FP32),
                 idx % 2 ? 1.25f : 1.f, idx % 5 == 1 ? 1.f : (idx % 4 == 3 ? 0.5f : 0.f)};
          fails += run_case(c); ++idx;
        }
  // larger problems (large-tile configuration)
  fails += run_case(Case{EFFMA_OP_T, EFFMA_OP_N, 1100, 1300, 512, EFFMA_E4M3, EFFMA_E4M3, EFFMA_BF16, -1, EFFMA_FP32, 1.f, 1.f});
  fails += run_case(Case{EFFMA_OP_N, EFFMA_OP_N, 1029, 1500, 256, EFFMA_E4M3, EFFMA_BF16, EFFMA_FP32, -1, -1, 0.5f, 1.f});
  // outer (row-wise) scaling: per row of op(A), per column of op(B), both; packed BF16 and generic stores
  int v = 1;
  for (effma_op ta : {EFFMA_OP_N, EFFMA_OP_T})
    for (effma_op tb : {EFFMA_OP_N, EFFMA_OP_T})
      for (effma_dtype dt : {EFFMA_BF16, EFFMA_FP32}) {
        Case c{ta, tb, 257, 129, 256, EFFMA_E4M3, EFFMA_E4M3, dt, dt == EFFMA_BF16 ? -1 : (int)EFFMA_BF16,
               dt == EFFMA_FP32 ? (int)EFFMA_FP32 : -1, 1.25f, dt == EFFMA_BF16 ? 0.f : 0.5f};
        c.vec = v; v = v % 3 + 1;
        fails += run_case(c);
      }
  {
    Case c{EFFMA_OP_T, EFFMA_OP_N, 1100, 1300, 512, EFFMA_E4M3, EFFMA_BF16, EFFMA_BF16, -1, -1, 1.f, 0.f};
    c.vec = 1;  // E4M3 weights per output row x BF16 activations (B's flag is ignored for BF16)
    fails += run_case(c);
  }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
