// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Exact reference model of one MAC: RN_BF16(ACC + exact(A*B)), A, B E4M3, ACC BF16 (host + device).
//
// Source of truth: integer arithmetic only, no host/device float.
// Every finite operand is an integer multiple of 2^-133 (the BF16 minimum
// subnormal; E4M3 products are multiples of 2^-18), so the exact sum
// ACC + A*B is held as a signed 320-bit integer N in units of 2^-133 and
// rounded to BF16 with round-to-nearest-even directly on that integer.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define EFFMA_REF_FN __host__ __device__ inline
#else
#define EFFMA_REF_FN inline
#endif

namespace effma_ref {

struct W5 { uint64_t w[5]; };  // little-endian limbs, magnitude only

EFFMA_REF_FN W5 w_zero() { W5 r; for (int i = 0; i < 5; ++i) r.w[i] = 0; return r; }

EFFMA_REF_FN W5 w_shl(uint32_t v, int s) {  // v << s, s in [0, 300)
  W5 r = w_zero();
  int li = s >> 6, bi = s & 63;
  uint64_t x = v;
  r.w[li] |= x << bi;
  if (bi && li + 1 < 5) r.w[li + 1] |= x >> (64 - bi);
  return r;
}
EFFMA_REF_FN int w_cmp(const W5 &a, const W5 &b) {
  for (int i = 4; i >= 0; --i)
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
  return 0;
}
EFFMA_REF_FN W5 w_add(const W5 &a, const W5 &b) {
  W5 r; uint64_t c = 0;
  for (int i = 0; i < 5; ++i) {
    uint64_t s = a.w[i] + b.w[i]; uint64_t c1 = s < a.w[i];
    r.w[i] = s + c; c = c1 | (r.w[i] < s);
  }
  return r;
}
EFFMA_REF_FN W5 w_sub(const W5 &a, const W5 &b) {  // requires a >= b
  W5 r; uint64_t br = 0;
  for (int i = 0; i < 5; ++i) {
    uint64_t d = a.w[i] - b.w[i]; uint64_t b1 = a.w[i] < b.w[i];
    r.w[i] = d - br; br = b1 | (d < br);
  }
  return r;
}
EFFMA_REF_FN int w_msb(const W5 &a) {  // -1 if zero
  for (int i = 4; i >= 0; --i)
    if (a.w[i]) { int k = 63; while (!((a.w[i] >> k) & 1)) --k; return i * 64 + k; }
  return -1;
}
EFFMA_REF_FN uint32_t w_bit(const W5 &a, int k) { return (uint32_t)((a.w[k >> 6] >> (k & 63)) & 1); }
EFFMA_REF_FN bool w_any_below(const W5 &a, int k) {  // any bit in [0, k) set
  for (int i = 0; i < 5 && i * 64 < k; ++i) {
    int n = k - i * 64;
    uint64_t m = n >= 64 ? ~0ull : ((1ull << n) - 1);
    if (a.w[i] & m) return true;
  }
  return false;
}

// value = (-1)^s * N * 2^-133
EFFMA_REF_FN W5 bf16_mag(uint32_t h) {
  uint32_t e = (h >> 7) & 0xFF, f = h & 0x7F;
  if (e == 0) return w_shl(f, 0);
  return w_shl(128 + f, (int)e - 1);
}
EFFMA_REF_FN bool e4m3_is_nan(uint32_t b) { return (b & 0x7F) == 0x7F; }
EFFMA_REF_FN W5 e4m3_prod_mag(uint32_t a, uint32_t b) {
  uint32_t ea = (a >> 3) & 15, eb = (b >> 3) & 15;
  uint32_t ma = (a & 7) | (ea ? 8 : 0), mb = (b & 7) | (eb ? 8 : 0);
  int ea1 = ea ? (int)ea : 1, eb1 = eb ? (int)eb : 1;
  // |E4M3| = m * 2^(e'-10); product = ma*mb * 2^(ea'+eb'-20) = ma*mb << (ea'+eb'+113) units
  return w_shl(ma * mb, ea1 + eb1 + 113);
}

// Round-to-nearest-even of (-1)^s * N * 2^-133 to BF16 bits.
EFFMA_REF_FN uint32_t round_bf16(uint32_t s, const W5 &N) {
  int k = w_msb(N);
  uint32_t sb = s << 15;
  if (k < 0) return sb;  // zero (sign decided by caller)
  if (k <= 7) {          // exactly representable: subnormal (N<128) or E=1
    uint32_t n = (uint32_t)N.w[0];
    return sb | n;       // encoding of N in [0,255] is N itself (E=n>>7, f=n&127)
  }
  int sh = k - 7;
  uint32_t sig = 0;
  for (int i = 0; i < 8; ++i) sig |= w_bit(N, sh + i) << i;
  uint32_t g = w_bit(N, sh - 1);
  bool st = w_any_below(N, sh - 1);
  if (g && (st || (sig & 1))) ++sig;
  if (sig == 256) { sig = 128; ++sh; }
  uint32_t E = (uint32_t)sh + 1;
  if (E >= 255) return sb | 0x7F80;  // overflow -> inf
  return sb | (E << 7) | (sig - 128);
}

// One lane: RN_BF16(acc + exact(a*b)).  a, b: E4M3 bytes; acc: BF16 bits.
EFFMA_REF_FN uint32_t lane(uint32_t a, uint32_t b, uint32_t acc) {
  const uint32_t QNAN = 0x7FC0;
  if (e4m3_is_nan(a) || e4m3_is_nan(b)) return QNAN;
  if ((acc & 0x7FFF) > 0x7F80) return QNAN;
  if ((acc & 0x7FFF) == 0x7F80) return acc;  // inf + finite
  uint32_t sa = acc >> 15, sp = ((a ^ b) >> 7) & 1;
  W5 Na = bf16_mag(acc), Np = e4m3_prod_mag(a, b);
  bool za = w_msb(Na) < 0, zp = w_msb(Np) < 0;
  if (za && zp) return (sa & sp) << 15;  // -0 + -0 = -0, otherwise +0
  if (sa == sp) return round_bf16(sa, w_add(Na, Np));
  int c = w_cmp(Na, Np);
  if (c == 0) return 0;  // exact cancellation -> +0 under RN
  return c > 0 ? round_bf16(sa, w_sub(Na, Np)) : round_bf16(sp, w_sub(Np, Na));
}

}  // namespace effma_ref
