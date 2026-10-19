// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E4M3EffMA core, exhaustively over 2^32 inputs each (src/e4m3effma.cuh), and effma_cast:
//  1. mac  : the kernels' MAC path -- decode E4M3 A, B; one FFMA onto the BF16 accumulator; RNE to
//            BF16 -- == RN_BF16(ACC + exact(A*B)) for every (A, B, ACC) = 2^32 cases.  Reference:
//            tests/e4m3effma_ref.h (exact integer model).  Domain: no E4M3 NaN code, no BF16 Inf/NaN
//            accumulator (out-of-domain inputs are mapped into the domain, so coverage stays complete).
//  2. rne  : FP32 -> BF16 rounding on every FP32 bit pattern vs an independent remainder-comparison
//            RNE (NaN -> 0x7FC0).
//  3. e4m3 : FP32 -> E4M3 encoding on every FP32 bit pattern vs an independent search for the nearest
//            finite E4M3 value (midpoint table, ties to the even code, saturation to +-448, every
//            NaN -> 0x7F).
//  4. effma_cast: 1M random FP32 bit patterns (incl. NaN / Inf) -> E4M3 / BF16 / FP16.
// Build with -DEFFMA_NEGATIVE_CONTROLS to check that deliberately broken variants are rejected.
#include "test_util.h"
#include "e4m3effma_ref.h"

struct Err { unsigned long long count; uint32_t x[16], y[16], got[16], exp[16]; };

__device__ __forceinline__ void record(Err *err, uint32_t x, uint32_t y, uint32_t got, uint32_t exp) {
  unsigned long long e = atomicAdd(&err->count, 1ull);
  if (e < 16) { err->x[e] = x; err->y[e] = y; err->got[e] = got; err->exp[e] = exp; }
}

// ---- implementations under test -------------------------------------------------------------
__device__ __forceinline__ float mac_f32(uint32_t a, uint32_t b, uint32_t acc) {
  return effma::fma_rn(effma::e4m3_value(a), effma::e4m3_value(b), __uint_as_float(acc << 16));
}
__device__ __forceinline__ uint32_t mac_ok(uint32_t a, uint32_t b, uint32_t acc) {
  return effma::rne_hi(__float_as_uint(mac_f32(a, b, acc))) >> 16;
}
__device__ __forceinline__ uint32_t rne_ok(uint32_t u) { return effma::rne_hi(u) >> 16; }
#ifdef EFFMA_NEGATIVE_CONTROLS
__device__ __forceinline__ uint32_t mac_trunc(uint32_t a, uint32_t b, uint32_t acc) {  // truncation, no RNE
  return __float_as_uint(mac_f32(a, b, acc)) >> 16;
}
__device__ __forceinline__ uint32_t mac_ftz(uint32_t a, uint32_t b, uint32_t acc) {  // flushes subnormal results
  const uint32_t r = mac_ok(a, b, acc);
  return (r & 0x7F80u) == 0 && (r & 0x7Fu) ? r & 0x8000u : r;
}
__device__ __forceinline__ uint32_t rne_nonan(uint32_t u) { return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
__device__ __forceinline__ uint32_t e4m3_ties_away(uint32_t u) {  // round half away instead of even
  const float x = __uint_as_float(u), a = fabsf(x);
  if (a != a || a >= 448.f || a < 0x1p-6f) return effma::f32_to_e4m3(x);
  const uint32_t b = __float_as_uint(a), r = b + 0x80000u;
  return ((u >> 24) & 0x80u) | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
}
#endif

// ---- kernels ---------------------------------------------------------------------------------
template <uint32_t (*OP)(uint32_t, uint32_t, uint32_t)>
__global__ void k_mac(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t i = (uint32_t)k;
    uint32_t a = i & 0xFF, b = (i >> 8) & 0xFF, c = i >> 16;
    if ((a & 0x7F) == 0x7F) a ^= 1;                           // E4M3 NaN -> neighbour
    if ((b & 0x7F) == 0x7F) b ^= 1;
    if ((c & 0x7F80) == 0x7F80) c = (c & 0x8000) | 0x7F7F;    // BF16 Inf / NaN -> max finite
    const uint32_t got = OP(a, b, c), exp = effma_ref::lane(a, b, c);
    if (got != exp) record(err, i, 0, got, exp);
  }
}

template <uint32_t (*OP)(uint32_t)>
__global__ void k_rne(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    uint32_t u = (uint32_t)k, exp;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) exp = 0x7FC0;  // NaN
    else {
      uint32_t hi = u >> 16, lo = u & 0xFFFF;
      exp = (lo > 0x8000 || (lo == 0x8000 && (hi & 1))) ? hi + 1 : hi;
    }
    uint32_t got = OP(u);
    if (got != exp) record(err, u, 0, got, exp);
  }
}

// Independent FP32 -> E4M3 reference: binary search over the midpoints between consecutive
// non-negative finite E4M3 values (midpoints have <= 5 significant bits: exact in FP32).
__constant__ float c_mid[126];  // c_mid[c] = (value(c) + value(c+1)) / 2, c = 0..125
__device__ __forceinline__ uint32_t e4m3_enc_ref(float x) {
  const uint32_t u = __float_as_uint(x), s = (u >> 24) & 0x80u;
  const float a = fabsf(x);
  if (a != a) return 0x7Fu;
  if (a >= 448.f) return s | 0x7Eu;
  int lo = 0, hi = 126;  // answer c in [0, 126]: first c with a < c_mid[c] (or == and c even)
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (a < c_mid[mid] || (a == c_mid[mid] && (mid & 1) == 0)) hi = mid; else lo = mid + 1;
  }
  return s | (uint32_t)lo;
}
__device__ __forceinline__ uint32_t e4m3_ok(uint32_t u) { return effma::f32_to_e4m3(__uint_as_float(u)); }

template <uint32_t (*OP)(uint32_t)>
__global__ void k_e4m3(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t u = (uint32_t)k, got = OP(u), exp = e4m3_enc_ref(__uint_as_float(u));
    if (got != exp) record(err, u, 0, got, exp);
  }
}

template <class LAUNCH>
static int run(const char *name, LAUNCH launch, uint64_t total) {
  Err *d; cudaMalloc(&d, sizeof(Err)); cudaMemset(d, 0, sizeof(Err));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const uint64_t chunk = 1ull << 28;
  for (uint64_t b = 0; b < total; b += chunk) launch(b, total - b < chunk ? total - b : chunk, d);
  cudaEventRecord(e1);
  cudaError_t ce = cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  Err h; cudaMemcpy(&h, d, sizeof(Err), cudaMemcpyDeviceToHost); cudaFree(d);
  if (ce) { printf("%-12s CUDA error %s\n", name, cudaGetErrorString(ce)); return 1; }
  printf("%-12s %s  cases=%llu  mismatches=%llu  (%.1f s)\n", name, h.count ? "FAIL" : "PASS",
         (unsigned long long)total, h.count, ms / 1e3);
  for (unsigned e = 0; e < h.count && e < 8; ++e)
    printf("   in=%08x %08x got=%08x exp=%08x\n", h.x[e], h.y[e], h.got[e], h.exp[e]);
  return h.count != 0;
}

#ifndef EFFMA_NEGATIVE_CONTROLS
// ---- effma_cast ------------------------------------------------------------------------------
static int test_cast() {
  int fails = 0;
  const size_t N = 1 << 20;
  std::vector<float> hx(N);
  for (size_t i = 0; i < N; ++i) { uint32_t u = rnd(); memcpy(&hx[i], &u, 4); }
  for (int i = 0; i < 256; ++i) hx[i] = e4((uint8_t)i);
  float *dx = dev(hx), *dsc = devf(0.5f);
  for (effma_dtype t : {EFFMA_E4M3, EFFMA_BF16, EFFMA_FP16}) {
    std::vector<uint8_t> out(N * esize(t)), exp(N * esize(t));
    uint8_t *dy; cudaMalloc(&dy, out.size());
    effma_tensor s = {dx, EFFMA_FP32, nullptr, nullptr, nullptr}, d = {dy, t, nullptr, dsc, nullptr};
    effma_status st = effma_cast(&s, &d, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dy, out.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; ++i) hstore(t, exp.data(), i, hx[i], 0.5f);
    size_t bad = 0; for (size_t i = 0; i < out.size(); ++i) bad += out[i] != exp[i];
    printf("cast FP32 -> %s (1M random bit patterns incl. NaN/Inf): %s\n", DN[t], (st == EFFMA_OK && !bad) ? "ok" : "FAIL");
    fails += st != EFFMA_OK || bad;
    cudaFree(dy);
  }
  cudaFree(dx); cudaFree(dsc);
  return fails;
}
#endif

#define MAC(OP) [](uint64_t b, uint64_t n, Err *d) { k_mac<OP><<<1024, 256>>>(b, n, d); }
#define RNE(OP) [](uint64_t b, uint64_t n, Err *d) { k_rne<OP><<<1024, 256>>>(b, n, d); }
#define ENC(OP) [](uint64_t b, uint64_t n, Err *d) { k_e4m3<OP><<<1024, 256>>>(b, n, d); }

int main(int argc, char **argv) {
  {  // midpoint table for the E4M3 encoder reference
    float val[127], mid[126];
    for (int c = 0; c < 127; ++c) {
      const int e = (c >> 3) & 15, f = c & 7;
      val[c] = e ? ldexpf(1.f + f / 8.f, e - 7) : ldexpf(f / 8.f, -6);
    }
    for (int c = 0; c < 126; ++c) mid[c] = 0.5f * (val[c] + val[c + 1]);
    cudaMemcpyToSymbol(c_mid, mid, sizeof mid);
  }
  uint64_t total = (argc > 1 && !strcmp(argv[1], "--quick")) ? 1ull << 26 : 1ull << 32;
#ifndef EFFMA_NEGATIVE_CONTROLS
  int fails = 0;
  fails += run("mac", MAC(mac_ok), total);
  fails += run("rne", RNE(rne_ok), total);
  fails += run("e4m3", ENC(e4m3_ok), total);
  fails += test_cast();
  printf(fails ? "\nFAILED\n" : "\nALL PASS\n");
  return fails != 0;
#else
  int caught = 0;
  caught += run("neg_trunc", MAC(mac_trunc), total);
  caught += run("neg_ftz", MAC(mac_ftz), total);
  caught += run("neg_rne_nan", RNE(rne_nonan), total);
  caught += run("neg_e4m3_away", ENC(e4m3_ties_away), total);
  printf(caught == 4 ? "\nall 4 broken variants rejected\n" : "\nNEGATIVE CONTROL NOT CAUGHT\n");
  return caught != 4;
#endif
}
