// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E4M3EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E4M3EffMA core (host side): effma_cast, effma_status_string and the choice of the kernel table
// for the current device.
#include <stdarg.h>
#include <stdio.h>
#include "e4m3effma.cuh"

namespace effma {

static thread_local char g_error[512];

void clear_error() { g_error[0] = 0; }
effma_status fail(effma_status s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, ap);
  va_end(ap);
  return s;
}
effma_status cuda_status(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return EFFMA_OK;
  return fail(EFFMA_CUDA_ERROR, "%s: CUDA error %d (%s): %s", what, (int)e, cudaGetErrorName(e), cudaGetErrorString(e));
}

namespace kern {

// Compiled with the same architectures as the kernels: tells which SASS the device actually runs.
__global__ void probe_kernel() {}

const DeviceInfo &device_info() {
  static DeviceInfo cache[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  DeviceInfo &d = cache[dev & 63];
  if (!d.table) {
    int ma = 0, mi = 0;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = ma * 10 + mi;
    // A tuned table only applies when the device runs SASS built for exactly its compute capability:
    // other architectures carry kernel bodies for their own table and the defaults only.
    cudaFuncAttributes fa;
    const bool loads = cudaFuncGetAttributes(&fa, probe_kernel) == cudaSuccess;
    if (!loads) cudaGetLastError();  // "no kernel image" is reported by check_device, not left sticky
    d.supported = loads && d.cc >= 50 && d.cc < 89;
    d.table = &kTables[loads && fa.binaryVersion == d.cc ? table_index(d.cc) : 0];
  }
  return d;
}

__global__ void cast_kernel(int tin, int tout, const void *__restrict__ src, void *__restrict__ dst,
                            size_t count, const float *sinv_p, const float *scale_p, float *amax) {
  const float sinv = scale_or_one(sinv_p), qs = scale_or_one(scale_p);
  float mx = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const float v = load_any(tin, src, i, sinv);
    mx = fmaxf(mx, fabsf(v));
    store_any(tout, dst, i, v, qs);
  }
  if (amax) amax_commit(amax, mx);
}

}  // namespace kern

effma_status check_device() {
  int dev = -1;
  const cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_status(e, "cudaGetDevice");
  const kern::DeviceInfo &d = kern::device_info();
  if (d.supported) return EFFMA_OK;
  if (d.cc >= 89)
    return fail(EFFMA_NOT_SUPPORTED, "device %d is sm_%d, which has native FP8: use cuBLASLt / Transformer Engine "
                "(E4M3EffMA targets sm_50..sm_87)", dev, d.cc);
  if (d.cc < 50) return fail(EFFMA_NOT_SUPPORTED, "device %d is sm_%d; E4M3EffMA needs sm_50 or newer", dev, d.cc);
  return fail(EFFMA_NOT_SUPPORTED, "device %d is sm_%d, but this build of the library has no code for it: "
              "rebuild with ARCHS including %d", dev, d.cc, d.cc);
}

}  // namespace effma

extern "C" const char *effma_last_error(void) { return effma::g_error; }

extern "C" effma_status effma_cast(const effma_tensor *src, const effma_tensor *dst, size_t count, cudaStream_t stream) {
  using effma::fail;
  effma::clear_error();
  const auto valid = [](effma_dtype t) { return t >= EFFMA_E4M3 && t <= EFFMA_FP32; };
  if (!src || !dst || !src->data || !dst->data) return fail(EFFMA_INVALID_VALUE, "effma_cast: src / dst or their data is NULL");
  if (!valid(src->dtype) || !valid(dst->dtype))
    return fail(EFFMA_INVALID_VALUE, "effma_cast: invalid dtype (src %d, dst %d)", (int)src->dtype, (int)dst->dtype);
  if (src->scaling != EFFMA_SCALING_TENSOR || dst->scaling != EFFMA_SCALING_TENSOR)
    return fail(EFFMA_INVALID_VALUE, "effma_cast: per-tensor scaling only");
  if (count == 0) return EFFMA_OK;
  if (effma_status st = effma::check_device()) return st;
  const int grid = (int)((count + 255) / 256 < 8192 ? (count + 255) / 256 : 8192);
  effma::kern::cast_kernel<<<grid, 256, 0, stream>>>(src->dtype, dst->dtype, src->data, dst->data, count,
                                                     src->dtype == EFFMA_E4M3 ? src->scale_inv : nullptr,
                                                     dst->dtype == EFFMA_E4M3 ? dst->scale : nullptr, dst->amax);
  return effma::cuda_status(cudaGetLastError(), "effma_cast: kernel launch");
}

extern "C" const char *effma_status_string(effma_status s) {
  switch (s) {
    case EFFMA_OK: return "EFFMA_OK";
    case EFFMA_INVALID_VALUE: return "EFFMA_INVALID_VALUE";
    case EFFMA_NOT_SUPPORTED: return "EFFMA_NOT_SUPPORTED";
    case EFFMA_CUDA_ERROR: return "EFFMA_CUDA_ERROR";
  }
  return "unknown effma_status";
}
