#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Independent check of the C++ reference with Python Fractions (exact rationals)."""
import ctypes, random, sys
from fractions import Fraction as F
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else 'out/libref.so')
lib.ref_lane.restype = ctypes.c_uint32
NAN = 'nan'

def e4m3(b):
    s, e, f = b >> 7, (b >> 3) & 15, b & 7
    if (b & 0x7F) == 0x7F: return NAN
    v = F(f, 8) * F(2) ** -6 if e == 0 else (1 + F(f, 8)) * F(2) ** (e - 7)
    return (s, v)

def bf16(h):
    s, e, f = h >> 15, (h >> 7) & 0xFF, h & 0x7F
    if e == 255: return NAN if f else ('inf', s)
    v = F(f, 128) * F(2) ** -126 if e == 0 else (1 + F(f, 128)) * F(2) ** (e - 127)
    return (s, v)

# all finite non-negative BF16 values, sorted, for rounding by search
POS = [(h, bf16(h)[1]) for h in range(0x7F80)]
import bisect
VALS = [v for _, v in POS]

def rn_bf16(x):  # x: Fraction, returns bits
    s = 1 if x < 0 else 0
    a = -x if x < 0 else x
    i = bisect.bisect_left(VALS, a)
    if i < len(VALS) and VALS[i] == a: return (s << 15) | POS[i][0]
    if i == len(VALS):  # above max finite
        lo = VALS[-1]; ulp = F(2) ** (127 - 7)
        mid = lo + ulp / 2  # tie goes to even = inf (max finite 0x7F7F is odd)
        return (s << 15) | (0x7F80 if a >= mid else 0x7F7F)
    lo, hi = VALS[i - 1], VALS[i]
    mid = (lo + hi) / 2
    if a < mid: h = POS[i - 1][0]
    elif a > mid: h = POS[i][0]
    else: h = POS[i - 1][0] if POS[i - 1][0] % 2 == 0 else POS[i][0]
    return (s << 15) | h

def model(a, b, acc):
    A, B, C = e4m3(a), e4m3(b), bf16(acc)
    if A == NAN or B == NAN or C == NAN: return 0x7FC0
    if C[0] == 'inf': return acc
    sp = A[0] ^ B[0]; p = A[1] * B[1]
    pv = -p if sp else p; cv = -C[1] if C[0] else C[1]
    x = cv + pv
    if x == 0:
        return 0x8000 if (p == 0 and C[1] == 0 and sp and C[0]) else 0
    return rn_bf16(x)

def check(a, b, acc):
    r, m = lib.ref_lane(a, b, acc), model(a, b, acc)
    if r != m:
        print(f'MISMATCH a={a:02x} b={b:02x} acc={acc:04x}: ref={r:04x} model={m:04x}'); return 1
    return 0

random.seed(1)
corpus = sorted(set([0, 0x8000, 1, 0x8001, 0x7F, 0x807F, 0x80, 0x8080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
           0x7FC0, 0xFFC0, 0x7F81, 0x3F80, 0xBF80, 0x3F81, 0xBF81, 0x3F7F, 0x4380, 0xC380, 0x3B80, 0x3480,
           0x4700, 0xC700, 0x4361, 0x3C00, 0x3800, 0x3700, 0x3680, 0x3600]
          + [random.randrange(0x10000) for _ in range(40)]))
bad = n = 0
# all E4M3 pairs x small acc corpus
for acc in corpus[:12]:
    for a in range(256):
        for b in range(256):
            bad += check(a, b, acc); n += 1
print('pairs x corpus12:', n, 'bad', bad)
# targeted: acc near the product (cancellation / ties): acc = rn(product) +- k ulps, both signs
for _ in range(300000):
    a, b = random.randrange(256), random.randrange(256)
    A, B = e4m3(a), e4m3(b)
    if A == NAN or B == NAN or A[1] * B[1] == 0: continue
    t = rn_bf16(A[1] * B[1]) + random.randint(-300, 300)
    acc = (t & 0x7FFF) | (random.randrange(2) << 15)
    bad += check(a, b, acc); n += 1
for _ in range(300000):
    bad += check(random.randrange(256), random.randrange(256), random.randrange(0x10000)); n += 1
print('total', n, 'bad', bad)
sys.exit(1 if bad else 0)
