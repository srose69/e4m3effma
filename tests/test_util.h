// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Host side of the E4M3EffMA tests: element helpers and a bit-exact emulation of effma_gemm in the
// documented operation order (include/e4m3effma.h), including the GEMV kernel's summation order.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/e4m3effma.h"
#include "../src/e4m3effma.cuh"  // f32_to_e4m3 (host + device)

static uint32_t rng = 0x2545F491u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frnd() { return (float)(rnd() & 0xFFFFFF) / 16777216.0f * 2.f - 1.f; }
static const char *DN[] = {"E4M3", "BF16", "FP16", "FP32"};

static float e4(uint8_t b) {
  int e = (b >> 3) & 15, f = b & 7;
  float v = e ? ldexpf(1.f + f / 8.f, e - 7) : ldexpf(f / 8.f, -6);
  return (b & 0x80) ? -v : v;
}
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t to_bf(float v) {
  uint32_t u; memcpy(&u, &v, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static size_t esize(effma_dtype t) { return t == EFFMA_E4M3 ? 1 : t == EFFMA_FP32 ? 4 : 2; }
static float hload(effma_dtype t, const void *p, size_t i, float sinv) {
  switch (t) {
    case EFFMA_E4M3: return e4(((const uint8_t *)p)[i]) * sinv;
    case EFFMA_BF16: return bf(((const uint16_t *)p)[i]);
    case EFFMA_FP16: return __half2float(((const __half *)p)[i]);
    default: return ((const float *)p)[i];
  }
}
static void hstore(effma_dtype t, void *p, size_t i, float v, float qs) {
  switch (t) {
    case EFFMA_E4M3: ((uint8_t *)p)[i] = (uint8_t)effma::f32_to_e4m3(v * qs); break;
    case EFFMA_BF16: ((uint16_t *)p)[i] = to_bf(v); break;
    case EFFMA_FP16: ((__half *)p)[i] = __float2half_rn(v); break;
    default: ((float *)p)[i] = v;
  }
}
static void fill(effma_dtype t, std::vector<uint8_t> &buf, float range) {
  for (size_t i = 0; i < buf.size() / esize(t); ++i) {
    if (t == EFFMA_E4M3) { uint8_t c; do c = rnd() & 0xFF; while ((c & 0x7F) == 0x7F); buf[i] = c; }
    else hstore(t, buf.data(), i, frnd() * range, 1.f);
  }
}
template <class T> static T *dev(const std::vector<T> &h) {
  T *d; cudaMalloc(&d, h.size() * sizeof(T) + 16); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
static float *devf(float v) { float *d; cudaMalloc(&d, 4); cudaMemcpy(d, &v, 4, cudaMemcpyHostToDevice); return d; }

// ---- effma_gemm ------------------------------------------------------------------------------
struct Case {
  effma_op ta, tb; int m, n, k;
  effma_dtype at, bt, dt; int ct;  // ct: C dtype, -1 = accumulate into D (C = NULL)
  int bias;                        // -1 none, else bias dtype
  float alpha, beta;
  int vec = 0;                     // outer scaling: bit 0 A (per row of op(A)), bit 1 B (per column of op(B))
};

// Runs effma_gemm on random data and compares D and amax bit for bit with the host emulation.
static int run_case(const Case &c) {
  const bool a_kc = c.ta == EFFMA_OP_T, b_kc = c.tb == EFFMA_OP_N;
  const int lda = ((a_kc ? c.k : c.m) + 15) / 16 * 16 + 16, ldb = ((b_kc ? c.k : c.n) + 15) / 16 * 16 + 16;
  const int ldd = c.m + 3, ldc = c.ct >= 0 ? c.m + 1 : ldd;
  const bool gemv = c.n <= 8 && a_kc && b_kc && c.at == EFFMA_E4M3;
  const effma_dtype ct = c.ct >= 0 ? (effma_dtype)c.ct : c.dt;
  std::vector<uint8_t> hA((size_t)lda * (a_kc ? c.m : c.k) * esize(c.at)), hB((size_t)ldb * (b_kc ? c.n : c.k) * esize(c.bt));
  std::vector<uint8_t> hC((size_t)ldc * c.n * esize(ct)), hD((size_t)ldd * c.n * esize(c.dt)), hbias(c.m * 4);
  fill(c.at, hA, 2.f); fill(c.bt, hB, 2.f); fill(ct, hC, 4.f);
  if (c.ct < 0) { fill(c.dt, hD, 4.f); hC = hD; } else memset(hD.data(), 0xA5, hD.size());
  if (c.bias >= 0) fill((effma_dtype)c.bias, hbias, 8.f);
  const float sa = 0.37f, sb = 1.9f, csinv = 0.75f, qs = 0.03125f;

  uint8_t *dA = dev(hA), *dB = dev(hB), *dC = dev(hC), *dD = dev(hD), *dbias = dev(hbias);
  // outer-scaling vectors: one factor per row of op(A) / column of op(B)
  std::vector<float> hsa(c.m), hsb(c.n);
  for (float &x : hsa) x = 0.25f + (rnd() & 1023) / 512.f;
  for (float &x : hsb) x = 0.25f + (rnd() & 1023) / 512.f;
  const bool va = (c.vec & 1) && c.at == EFFMA_E4M3, vb = (c.vec & 2) && c.bt == EFFMA_E4M3;
  float *dsa = va ? dev(hsa) : devf(sa), *dsb = vb ? dev(hsb) : devf(sb);
  float *dcs = devf(csinv), *dqs = devf(qs), *damax = devf(0.f);
  effma_tensor A = {dA, c.at, dsa, nullptr, nullptr, va ? EFFMA_SCALING_OUTER : EFFMA_SCALING_TENSOR};
  effma_tensor B = {dB, c.bt, dsb, nullptr, nullptr, vb ? EFFMA_SCALING_OUTER : EFFMA_SCALING_TENSOR};
  effma_tensor C = {dC, ct, dcs, nullptr, nullptr}, D = {dD, c.dt, dcs, dqs, damax};
  effma_tensor bias = {dbias, c.bias >= 0 ? (effma_dtype)c.bias : EFFMA_FP32, nullptr, nullptr, nullptr};
  effma_status st = effma_gemm(c.ta, c.tb, c.m, c.n, c.k, c.alpha, &A, lda, &B, ldb, c.beta,
                               c.ct >= 0 ? &C : nullptr, ldc, &D, ldd, c.bias >= 0 ? &bias : nullptr, 0);
  cudaError_t ce = cudaDeviceSynchronize();
  std::vector<uint8_t> got(hD.size()); float gamax;
  cudaMemcpy(got.data(), dD, got.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gamax, damax, 4, cudaMemcpyDeviceToHost);

  auto Aval = [&](int i, int kk) { return hload(c.at, hA.data(), a_kc ? (size_t)i * lda + kk : (size_t)kk * lda + i, 1.f); };
  auto Bval = [&](int kk, int j) { return hload(c.bt, hB.data(), b_kc ? (size_t)j * ldb + kk : (size_t)kk * ldb + j, 1.f); };
  auto scale = [&](int i, int j) {  // (alpha * sa[i]) * sb[j], as the epilogue
    const float a = c.at == EFFMA_E4M3 ? (va ? hsa[i] : sa) : 1.f, b = c.bt == EFFMA_E4M3 ? (vb ? hsb[j] : sb) : 1.f;
    return (c.alpha * a) * b;
  };
  std::vector<uint8_t> exp = hD;
  float eamax = 0.f;
  for (int j = 0; j < c.n; ++j)
    for (int i = 0; i < c.m; ++i) {
      float acc = 0.f;
      if (!gemv) {
        for (int kk = 0; kk < c.k; ++kk) acc = fmaf(Aval(i, kk), Bval(kk, j), acc);
      } else {
        float lanes[32];
        for (int l = 0; l < 32; ++l) {
          float p = 0.f;
          for (int kb = 8 * l; kb < c.k; kb += 256)
            for (int jj = 0; jj < 8; ++jj) p = fmaf(Aval(i, kb + jj), Bval(kb + jj, j), p);
          lanes[l] = p;
        }
        for (int o = 16; o; o >>= 1) {
          float nx[32];
          for (int l = 0; l < 32; ++l) nx[l] = lanes[l] + lanes[l ^ o];
          memcpy(lanes, nx, sizeof nx);
        }
        acc = lanes[0];
      }
      float v = acc * scale(i, j);
      if (c.bias >= 0) v = v + hload((effma_dtype)c.bias, hbias.data(), i, 1.f);
      if (c.beta != 0.f) v = fmaf(c.beta, hload(ct, hC.data(), (size_t)j * ldc + i, csinv), v);
      eamax = fmaxf(eamax, fabsf(v));
      hstore(c.dt, exp.data(), (size_t)j * ldd + i, v, qs);
    }
  size_t bad = 0;
  for (size_t i = 0; i < exp.size(); ++i) bad += exp[i] != got[i];
  const bool amax_ok = gamax == eamax;
  const int fail = st != EFFMA_OK || ce != cudaSuccess || bad || !amax_ok;
  printf("%s %c%c m=%4d n=%4d k=%5d A=%s B=%s D=%s C=%-5s bias=%-4s a=%.2f b=%5.2f%s : %s%s\n",
         gemv ? "gemv" : "gemm", c.ta == EFFMA_OP_T ? 'T' : 'N', c.tb == EFFMA_OP_T ? 'T' : 'N', c.m, c.n, c.k,
         DN[c.at], DN[c.bt], DN[c.dt], c.ct >= 0 ? DN[c.ct] : "accum", c.bias >= 0 ? DN[c.bias] : "-", c.alpha,
         c.beta, va || vb ? (va && vb ? " rowwise A+B" : va ? " rowwise A" : " rowwise B") : "", fail ? "FAIL" : "ok",
         amax_ok ? "" : " (amax mismatch)");
  if (bad) printf("   %zu bytes differ\n", bad);
  if (st != EFFMA_OK) printf("   status %s\n", effma_status_string(st));
  if (ce) printf("   cuda %s\n", cudaGetErrorString(ce));
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dbias);
  cudaFree(dsa); cudaFree(dsb); cudaFree(dcs); cudaFree(dqs); cudaFree(damax);
  return fail;
}

