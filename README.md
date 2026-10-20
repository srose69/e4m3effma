<div align="center">

# E4M3EffMA

**FP8 (E4M3) GEMM, GEMV, casts and fused AdamW for NVIDIA GPUs without native FP8**

![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia)
![Targets](https://img.shields.io/badge/targets-sm__50%E2%80%93sm__87-blue)
![Tested](https://img.shields.io/badge/tested%20on-GTX%201080%20%2B%20RTX%203090%20(community)-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![API](https://img.shields.io/badge/API-C%20%2F%20cuBLASLt--style-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0%20%2B%20linking%20exception-blue)

</div>

NVIDIA's native FP8 compute stack (cuBLASLt FP8, Transformer Engine, `torch._scaled_mm`) starts at
Ada / Hopper-class GPUs (sm_89+). **E4M3EffMA** brings the same data format and the same GEMM
semantics to the GPUs most people still have — Maxwell, Pascal, Volta, Turing, Ampere and Jetson Orin
(sm_50 … sm_87). Weights and activations stay in E4M3 (4× smaller than FP32), products are exact,
accumulation is FP32, and outputs come out in BF16 / FP16 / FP32 / E4M3.

> [!IMPORTANT]
> **Tested and tuned by the author on a GTX 1080 only.** The author has no GPU newer than Pascal;
> everything else is up to the community, or to time. So far an RTX 3090 (sm_86, CUDA 13.3) has
> passed the full test suite in a community run, untuned. The library compiles for every
> architecture from sm_50 to sm_87, but it runs the same FP32-FFMA kernels everywhere: it does **not**
> use Tensor Cores, `cp.async`, `ldmatrix`, or any other Volta/Turing/Ampere-specific feature, although
> it could and should on those GPUs. Contributions are very welcome — and so is hardware, which is
> honestly the better way to get those kernels written :)  See [Contributing](#contributing).

## Performance at a glance

GTX 1080. The baseline is cuBLAS on **FP32 data** of the same shape (SGEMM / SGEMV) — the fastest
way to run the same model on this GPU without FP8 — and PyTorch's fused AdamW for the optimizer.
Median times; the FP32 peak of each GEMM row is taken at the SM clock measured while that row ran:

| Workload | E4M3EffMA | Utilization | Baseline: cuBLAS FP32 | Speed vs baseline |
|---|---:|---|---:|---:|
| GEMM 4096³, E4M3 × E4M3 → BF16 | 19.5 ms | **77 %** of FP32 peak | SGEMM 19.2 ms (76 % of FP32 peak) | 0.98× |
| GEMM 4096³, E4M3 weights × BF16 activations | 18.7 ms | **79 %** of FP32 peak | SGEMM 19.2 ms (76 % of FP32 peak) | **1.03×** |
| GEMV 8192 × 8192, batch 1 | 0.237 ms | 283 GB/s, **88 %** of DRAM peak | SGEMV 1.041 ms | **4.4×** |
| GEMV 8192 × 8192, batch 8 (BF16 x) | 0.399 ms | 168 GB/s, 52 % of DRAM peak | SGEMM 1.537 ms | **3.9×** |
| AdamW, 64 M parameters, FP32 params + state | 7.43 ms | 253 GB/s, **79 %** of DRAM peak | `torch.optim.AdamW(fused=True)` 9.33 ms | **1.25×** |
| AdamW, 64 M parameters, BF16 params + state | 3.34 ms | 281 GB/s, **88 %** of DRAM peak | `torch.optim.AdamW(fused=True)` 8.99 ms | **2.69×** |

FP32 peak: 20 SMs × 128 FFMA × SM clock (1786–1847 MHz in these rows); DRAM peak: 320 GB/s. Per clock the
FP8 GEMM matches cuBLAS SGEMM; the E4M3 × E4M3 kernel draws more power (up to 215 W) and runs at a
slightly lower boost clock, the mixed one does not. On Pascal an exact FP8 MAC costs one FP32 FFMA, so
compute-bound GEMM tops out at SGEMM speed while reading 4× fewer bytes; bandwidth-bound kernels
(decoding, optimizer) get the full benefit of the smaller weights. CUTLASS has not been measured.

Results by GPU: [benches/table.md](benches/table.md) (full GTX 1080 run: [benches/1080.md](benches/1080.md);
RTX 3090, community: [benches/rtx-3090.md](benches/rtx-3090.md));
methodology and how to add your GPU: [docs/performance.md](docs/performance.md).

## Features

- **Exact products, FP32 accumulation.** Every finite E4M3 × E4M3 product is exact in FP32; one FFMA
  per MAC in a fixed order, one rounding at the end — deterministic and bit-reproducible for a fixed
  kernel path. Why not BF16 accumulation: [docs/numerics.md](docs/numerics.md).
- **cuBLASLt / Transformer Engine semantics.** `D = α·op(A)·op(B) + bias + β·C`, per-tensor or
  row-wise (per output row / column) `scale_inv` for FP8 inputs, `scale` + `amax` for FP8 outputs,
  TE-style accumulation into an FP32 `main_grad`.
- **Mixed operands.** A and B can each be E4M3, BF16 or FP32 — FP8 weights with BF16 or FP32
  activations.
- **GEMV path for decoding**, picked automatically for up to 8 columns, reading each weight byte once.
- **Fused Adam / AdamW** on FP32 or BF16 state, with an optional E4M3 copy of the updated weights.
- **PyTorch bindings**: functional API, a `torch.optim` AdamW, `torch.e4m3effma`,
  `torch._scaled_mm` working on pre-Ada GPUs, and `Float8Tensor`, an FP8 tensor that carries its scale,
  so `@`, `F.linear` and `nn.Linear` run FP8 matmuls unchanged.
- **Strict input checking** with readable errors in C (`effma_last_error`) and Python (`TypeError` /
  `ValueError` / `NotImplementedError`).
- **Per-architecture tuning** (`make tune`) and a 25-second build for all 12 architectures.
- **Heavily tested**: bit-exact against host emulation, exhaustive 2³² checks of the core arithmetic.

## Supported GPUs

| Architecture | Compute capability | Status |
|---|---|---|
| Pascal (GTX 10xx, Tesla P4/P40/P100) | sm_60, sm_61, sm_62 | ✅ tested and tuned by the author (GTX 1080, sm_61) |
| Maxwell | sm_50, sm_52, sm_53 | ⚙️ compiles, untested |
| Volta / Turing | sm_70, sm_72, sm_75 | ⚙️ compiles, untested, no Tensor Core path yet |
| Ampere / Orin | sm_80, sm_86, sm_87 | ✅ tests pass on RTX 3090 (sm_86, community); untuned, no Tensor Core / `cp.async` path yet |
| Ada, Hopper, Blackwell | sm_89+ | ❌ excluded on purpose — native FP8, use cuBLASLt / TE / `torch._scaled_mm` |

## Installation

**Python (PyTorch ≥ 2.1, CUDA toolkit of the same major version as PyTorch's):**

```sh
git clone https://github.com/srose69/e4m3effma && cd e4m3effma
pip install ./binds/python --no-build-isolation          # builds for the GPUs visible right now
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ./binds/python --no-build-isolation   # or chosen ones
```

**C / C++ (CUDA 12.0+, GNU make):**

```sh
make lib                                   # out/libe4m3effma.a for all of sm_50 … sm_87
make install PREFIX=$HOME/.local           # header + library + pkg-config (ARCHS="61 75" to narrow)
cc app.c $(pkg-config --cflags --libs e4m3effma)
```

Build options, the parallel build and tuning: [docs/building.md](docs/building.md).

## Usage

**PyTorch**

```python
import torch, e4m3effma as fx

x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
w = (torch.randn(4096, 4096, device="cuda") * 16).to(torch.e4m3effma)  # = torch.float8_e4m3fn
w_scale_inv = torch.tensor([1 / 16], device="cuda")

y = fx.linear(x, w, scale_w=w_scale_inv)                     # BF16 [8, 4096], GEMV kernel
y = torch._scaled_mm(x.to(torch.e4m3effma), w.t(),           # PyTorch's own FP8 matmul, now on sm_61
                     torch.tensor(1.0, device="cuda"), w_scale_inv[0], out_dtype=torch.bfloat16)

mlp = torch.nn.Sequential(torch.nn.Linear(4096, 4096), torch.nn.GELU()).cuda().bfloat16()
fx.quantize_linear_(mlp)                                      # weights -> fx.Float8Tensor, per-row scales
y = mlp(x)                                                    # FP8 x BF16 matmuls on E4M3EffMA

model = torch.nn.Linear(4096, 4096, device="cuda")
opt = fx.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, state_dtype=torch.bfloat16)
```

More (`matmul`, `cast`, `adam_step`, `Float8Tensor`, FP8 outputs with `amax`, error behaviour):
[binds/python/README.md](binds/python/README.md).

**C**

```c
#include "e4m3effma.h"

// Y[out, batch] (BF16) = W[out, in] (E4M3) · X[in, batch] (BF16), column-major TN as in cuBLASLt
effma_tensor W = {d_w, EFFMA_E4M3, d_w_scale_inv, NULL, NULL};   // data, dtype, scale_inv, scale, amax
effma_tensor X = {d_x, EFFMA_BF16, NULL, NULL, NULL};
effma_tensor Y = {d_y, EFFMA_BF16, NULL, NULL, NULL};
effma_status st = effma_gemm(EFFMA_OP_T, EFFMA_OP_N, out, batch, in, 1.0f, &W, in, &X, in,
                             0.0f, NULL, 0, &Y, out, /*bias=*/NULL, stream);
if (st != EFFMA_OK) fprintf(stderr, "%s\n", effma_last_error());   // e.g. "k = 24 is not a multiple of 16"
```

Full API, constraints and more examples: [docs/c-api.md](docs/c-api.md).

## Documentation

| | |
|---|---|
| [docs/c-api.md](docs/c-api.md) | C API, constraints, error handling, examples |
| [binds/python/README.md](binds/python/README.md) | Python / PyTorch bindings, `torch._scaled_mm` integration, `Float8Tensor` |
| [docs/building.md](docs/building.md) | build options, parallel build, install, tuning, project layout |
| [benches/table.md](benches/table.md) | benchmark results by GPU (one file per GPU in `benches/`) |
| [docs/performance.md](docs/performance.md) | how the benchmarks are run and measured, how to add a GPU, GTX 1080 analysis |
| [docs/numerics.md](docs/numerics.md) | exactness, accuracy, why FP32 accumulation and one rounding at the end, why BF16 |
| [docs/design.md](docs/design.md) | why one FFMA per MAC, packed half-registers, kernel configurations |
| [docs/testing.md](docs/testing.md) | what the tests check |
| [NOTICE](NOTICE), [LICENSE](LICENSE), [LICENSE-EXCEPTION](LICENSE-EXCEPTION) | license: AGPL v3 + linking and attribution exception |

## Roadmap

Things that would make the library much faster on newer GPUs, none of which can be tested here:

- [ ] **Tensor Core path for sm_70+.** E4M3 converts exactly to FP16 (sm_70+) and to BF16 (sm_80+, the
      native match for BF16 activations), so `mma.sync` with FP32 accumulation preserves exact E4M3
      products while moving the MAC work onto Tensor Cores, with potentially much higher throughput
      than scalar FFMA.
- [ ] **`cp.async` + multi-stage pipelines for sm_80+**, and `ldmatrix` for operand fetch.
- [ ] **Packed two-lane MACs on sm_70+** (separate INT32 pipe, fast HFMA2 / BF16x2) — not worth it on
      Pascal, possibly worth it there ([docs/design.md](docs/design.md)).
- [ ] **Per-architecture tuning entries** for Maxwell, Volta, Turing and Ampere GPUs.

## Contributing

Issues and pull requests are welcome, especially from anyone who can run the code on something newer
than a GTX 1080:

1. Run `make test` (and `pytest binds/python/tests`) on your GPU and report the result with the GPU model.
2. Run `make tune` and open a pull request with your entry in `src/e4m3effma_tuned.h`.
3. Run `bench/run_bench.sh` and open a pull request with the `benches/<gpu>.md` file and the
   `benches/table.md` row it writes (see [docs/performance.md](docs/performance.md)).
4. Tensor Core / `cp.async` kernels for sm_70+ / sm_80+ are the most valuable contribution. They must
   pass `make test` bit-exactly, or document why a different accumulation order is needed.

If you would rather donate or lend a newer GPU (Turing, Ampere or Jetson Orin), please open an issue —
that is the fastest way for those kernels to get written and tested.

Contributions are accepted under the project's license (below).

## License

E4M3EffMA is licensed under the [GNU AGPL v3](LICENSE) with the
[Linking and Attribution Exception](LICENSE-EXCEPTION); details and the exact attribution text are in
[NOTICE](NOTICE).

- **Use it in anything.** Programs and services that use E4M3EffMA — link it, call it, load its Python
  bindings — can be under any license, including proprietary ones.
- **Changes to E4M3EffMA itself stay open**: modified versions remain under the AGPL with the exception,
  with their source available, and must be marked as modified.
- **Credit is required**: every work that uses E4M3EffMA shows the attribution from NOTICE (for example
  in its third-party notices, credits or README).
- **A request, not a condition**: if you train or fine-tune a model with E4M3EffMA, please mention it in
  the model card or paper. [CITATION.cff](CITATION.cff) has the citation.
