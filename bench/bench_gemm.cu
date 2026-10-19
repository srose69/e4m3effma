// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// effma_gemm throughput (E4M3 x E4M3 -> BF16, mixed operands, FP32 accumulation) vs cuBLAS SGEMM of
// the same shape (FP32 in/out).  Median time per row; % of FP32 peak = SMs x 128 FFMA x the SM clock
// measured during that very row (see bench_util.h).
#include <vector>
#include <cublas_v2.h>
#include "../include/e4m3effma.h"
#include "bench_util.h"

int main(int argc, char **argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 4096, M = S, N = S, K = S;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s, %d SMs; M=N=K=%d; median of %d runs per row after %d s idle; %% of FP32 peak at the row's SM clock\n\n",
         p.name, p.multiProcessorCount, S, Bench::env("EFFMA_BENCH_REPS", 20), Bench::env("EFFMA_BENCH_COOL", 2));
  printf("%-50s %9s %13s %9s %12s\n", "", "time", "", "SM clock", "of FP32 peak");
  Bench bench;

  std::vector<unsigned char> h((size_t)M * K);
  unsigned r = 1;
  for (auto &x : h) { do { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x = r & 0xFF; } while ((x & 0x7F) == 0x7F); }
  void *dA, *dB, *dD; cudaMalloc(&dA, (size_t)M * K); cudaMalloc(&dB, (size_t)K * N); cudaMalloc(&dD, (size_t)M * N * 2);
  cudaMemcpy(dA, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(dB, h.data(), h.size(), cudaMemcpyHostToDevice);
  effma_tensor A = {dA, EFFMA_E4M3, nullptr, nullptr, nullptr}, B = {dB, EFFMA_E4M3, nullptr, nullptr, nullptr};
  effma_tensor D = {dD, EFFMA_BF16, nullptr, nullptr, nullptr};
  const double macs = (double)M * N * K;
  auto row = [&](const char *name, BenchRow x) {
    const double gmacs = macs / x.ms / 1e6, peak = p.multiProcessorCount * 128 * x.mhz / 1e3;
    if (x.mhz > 0) printf("%-50s %6.3f ms %7.1f GMAC/s %5.0f MHz %10.1f%%\n", name, x.ms, gmacs, x.mhz, 100 * gmacs / peak);
    else printf("%-50s %6.3f ms %7.1f GMAC/s %9s %11s\n", name, x.ms, gmacs, "n/a", "n/a");
    fflush(stdout);
  };
  const effma_op ops[][2] = {{EFFMA_OP_T, EFFMA_OP_N}, {EFFMA_OP_N, EFFMA_OP_N}, {EFFMA_OP_N, EFFMA_OP_T}, {EFFMA_OP_T, EFFMA_OP_T}};
  for (auto &o : ops) {
    const int lda = o[0] == EFFMA_OP_T ? K : M, ldb = o[1] == EFFMA_OP_N ? K : N;
    effma_status st = EFFMA_OK;
    const BenchRow x = bench.run([&] { st = effma_gemm(o[0], o[1], M, N, K, 1.f, &A, lda, &B, ldb, 0.f, nullptr, 0, &D, M, nullptr, 0); }, 20);
    char name[64]; snprintf(name, 64, "effma_gemm %c%c E4M3 -> BF16%s", o[0] == EFFMA_OP_T ? 'T' : 'N', o[1] == EFFMA_OP_T ? 'T' : 'N',
                            st == EFFMA_OK ? "" : " (FAILED)");
    row(name, x);
  }
  {  // mixed operands (E4M3 weights x BF16 / FP32 activations) and TE-style FP32 accumulation
    void *dX; cudaMalloc(&dX, (size_t)K * N * 4); cudaMemset(dX, 0, (size_t)K * N * 4);
    void *dF; cudaMalloc(&dF, (size_t)M * N * 4); cudaMemset(dF, 0, (size_t)M * N * 4);
    effma_tensor F = {dF, EFFMA_FP32, nullptr, nullptr, nullptr};
    for (effma_dtype bt : {EFFMA_BF16, EFFMA_FP32}) {
      effma_tensor X = {dX, bt, nullptr, nullptr, nullptr};
      char name[64]; snprintf(name, 64, "effma_gemm TN E4M3 x %s -> BF16", bt == EFFMA_BF16 ? "BF16" : "FP32");
      row(name, bench.run([&] { effma_gemm(EFFMA_OP_T, EFFMA_OP_N, M, N, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &D, M, nullptr, 0); }, 20));
    }
    effma_tensor X = {dX, EFFMA_BF16, nullptr, nullptr, nullptr};
    row("effma_gemm TN E4M3 x BF16, FP32 D += (accumulate)",
        bench.run([&] { effma_gemm(EFFMA_OP_T, EFFMA_OP_N, M, N, K, 1.f, &A, K, &X, K, 1.f, nullptr, 0, &F, M, nullptr, 0); }, 20));
    cudaFree(dX); cudaFree(dF);
  }
  float *fA, *fB, *fC; cudaMalloc(&fA, (size_t)M * K * 4); cudaMalloc(&fB, (size_t)K * N * 4); cudaMalloc(&fC, (size_t)M * N * 4);
  cudaMemset(fA, 0, (size_t)M * K * 4); cudaMemset(fB, 0, (size_t)K * N * 4);
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1, zero = 0;
  row("cuBLAS SGEMM NN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, fA, M, fB, K, &zero, fC, M); }, 20));
  row("cuBLAS SGEMM TN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, fA, K, fB, K, &zero, fC, M); }, 20));
  return 0;
}
