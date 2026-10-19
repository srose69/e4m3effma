// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// effma_adam, bit-exact against a host emulation of the documented order: Adam and AdamW, param
// FP32 / BF16 x grad FP32 / BF16 / FP16 x moments FP32 / BF16, with the E4M3 weight copy and amax;
// 100003 elements (a tail after the 8-element chunks) and an unaligned case (element-by-element path).
#include "test_util.h"

// unaligned: the gradient starts one element past a 16-byte boundary (element-by-element path)
static int test_adam(effma_dtype pt, effma_dtype gt, effma_dtype mt, int adamw, int step, bool unaligned = false) {
  const size_t N = 100003;
  effma_adam_config cfg = {1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, step, adamw, 1, 0.5f};
  std::vector<uint8_t> hp(N * esize(pt)), hg(N * esize(gt)), hm(N * esize(mt)), hv(N * esize(mt));
  fill(pt, hp, 1.f); fill(gt, hg, 0.1f); fill(mt, hm, 0.01f);
  for (size_t i = 0; i < N; ++i) hstore(mt, hv.data(), i, fabsf(hload(mt, hm.data(), i, 1.f)) * 0.01f, 1.f);
  uint8_t *dp = dev(hp), *dgb, *dm = dev(hm), *dv = dev(hv), *dq; cudaMalloc(&dq, N);
  cudaMalloc(&dgb, hg.size() + 16);
  uint8_t *dg = dgb + (unaligned ? esize(gt) : 0);
  cudaMemcpy(dg, hg.data(), hg.size(), cudaMemcpyHostToDevice);
  float *dqs = devf(64.f), *dam = devf(0.f);
  effma_tensor P = {dp, pt, nullptr, nullptr, nullptr}, G = {dg, gt, nullptr, nullptr, nullptr};
  effma_tensor Mo = {dm, mt, nullptr, nullptr, nullptr}, V = {dv, mt, nullptr, nullptr, nullptr};
  effma_tensor Q = {dq, EFFMA_E4M3, nullptr, dqs, dam};
  effma_status st = effma_adam(&cfg, N, &P, &G, &Mo, &V, &Q, 0);
  cudaDeviceSynchronize();
  std::vector<uint8_t> gp(hp.size()), gm(hm.size()), gv(hv.size()), gq(N); float gam;
  cudaMemcpy(gp.data(), dp, gp.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gm.data(), dm, gm.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(gv.data(), dv, gv.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gq.data(), dq, N, cudaMemcpyDeviceToHost);
  cudaMemcpy(&gam, dam, 4, cudaMemcpyDeviceToHost);
  // host emulation of the documented order
  const float omb1 = (float)(1.0 - (double)cfg.beta1), omb2 = (float)(1.0 - (double)cfg.beta2);
  const float rbc1 = (float)(1.0 / (1.0 - pow((double)cfg.beta1, step))), rbc2 = (float)(1.0 / (1.0 - pow((double)cfg.beta2, step)));
  const float lr_wd = (float)((double)cfg.lr * cfg.weight_decay);
  std::vector<uint8_t> ep = hp, em = hm, ev = hv, eq(N); float eam = 0.f;
  for (size_t i = 0; i < N; ++i) {
    float g = hload(gt, hg.data(), i, 1.f) * cfg.grad_scale, p = hload(pt, hp.data(), i, 1.f);
    if (!adamw) g = fmaf(cfg.weight_decay, p, g);
    const float m = fmaf(cfg.beta1, hload(mt, hm.data(), i, 1.f), omb1 * g);
    const float v = fmaf(cfg.beta2, hload(mt, hv.data(), i, 1.f), omb2 * (g * g));
    const float u = (m * rbc1) / (sqrtf(v * rbc2) + cfg.eps);
    if (adamw) p = fmaf(-lr_wd, p, p);
    p = fmaf(-cfg.lr, u, p);
    hstore(pt, ep.data(), i, p, 1.f); hstore(mt, em.data(), i, m, 1.f); hstore(mt, ev.data(), i, v, 1.f);
    eq[i] = (uint8_t)effma::f32_to_e4m3(p * 64.f); eam = fmaxf(eam, fabsf(p));
  }
  const bool ok = st == EFFMA_OK && gp == ep && gm == em && gv == ev && gq == eq && gam == eam;
  printf("adam%s step=%d param=%s grad=%s moments=%s +E4M3 copy%s: %s\n", adamw ? "w" : " ", step, DN[pt], DN[gt], DN[mt],
         unaligned ? ", unaligned grad" : "", ok ? "ok" : "FAIL");
  cudaFree(dp); cudaFree(dgb); cudaFree(dm); cudaFree(dv); cudaFree(dq); cudaFree(dqs); cudaFree(dam);
  return !ok;
}

int main() {
  int fails = 0;
  for (effma_dtype pt : {EFFMA_FP32, EFFMA_BF16})
    for (effma_dtype gt : {EFFMA_FP32, EFFMA_BF16, EFFMA_FP16})
      for (effma_dtype mt : {EFFMA_FP32, EFFMA_BF16}) fails += test_adam(pt, gt, mt, (pt + gt + mt) & 1, 1 + (gt * 3 + mt));
  fails += test_adam(EFFMA_FP32, EFFMA_BF16, EFFMA_FP32, 1, 3, true);
  fails += test_adam(EFFMA_BF16, EFFMA_FP16, EFFMA_BF16, 0, 4, true);
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
