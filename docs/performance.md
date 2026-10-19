# Performance

Results per GPU are in [benches/table.md](../benches/table.md) (one row per GPU) and in
`benches/<gpu>.md` (full tables and the conditions of each run). This page describes how they are
measured, how to add a GPU, and what the GTX 1080 numbers say.

## Running the benchmarks

```sh
bench/run_bench.sh                                          # GPU 0
GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
```

The script builds the benchmarks, sets the board up (below), runs `bench_gemm`, `bench_gemv` and
`bench_adamw`, and writes `benches/<gpu>.md` plus that GPU's row in `benches/table.md`
(`bench/report.py`). Raw output, including clock / temperature / power logs, stays in
`out/bench_<gpu>/`. It needs `sudo` for `nvidia-smi` and restores every setting on exit.
`make bench` runs the same benchmarks without touching the board settings. With PyTorch and the
bindings importable (`PYTHON=...`), the script also compares AdamW with `torch.optim.AdamW`
(`bench/bench_torch_adamw.py`).

**Adding a GPU:** run the script on it and open a pull request with `benches/<gpu>.md` and the
updated `benches/table.md`. Results from any pre-FP8 GPU are welcome (sm_50 … sm_87); please describe
the cooling in `COOLING`.

## Methodology

- **Median per row.** Every row (one kernel, one shape) gets one untimed call, then 2 s of idle GPU
  (`EFFMA_BENCH_COOL`), then 50–2000 timed calls (`EFFMA_BENCH_REPS`); the median is reported. The
  benchmarks are separated by 10 s of idle GPU and there is no warm-up, so every row starts from a
  similar temperature.
- **FP32 peak at the row's own clock.** % of FP32 peak = GMAC/s / (SMs × 128 × SM clock), one FFMA = one
  MAC, with the SM clock sampled through NVML every 5 ms while the row runs (`bench/bench_util.h`).
  GPU Boost moves the clock with temperature and power, and kernels draw different power, so one clock
  for the whole run would credit some rows with a clock they never had.
- **Board settings.** The power limit is raised to its maximum and the SM clock is allowed up to its
  top bin; the memory clock is locked at its maximum. Without a display, GeForce drivers run CUDA work
  in the P2 state with a reduced memory clock (4513 instead of 5005 MHz on a GTX 1080, about 10 % of
  bandwidth); the lock avoids that. The per-GPU file records what was actually set and the clock,
  temperature, power and throttling seen during each benchmark.
- **Bandwidth.** DRAM peak = memory clock × bus width; a device-to-device copy (read + write) is
  measured as the practical ceiling for streaming kernels.
- **Baselines.** cuBLAS on FP32 data of the same shape: SGEMM for GEMM, SGEMV and SGEMM n = 8 for the
  GEMV path. CUTLASS is not measured.

## What the GTX 1080 numbers say

Full tables: [benches/1080.md](../benches/1080.md).

- **GEMM matches cuBLAS SGEMM per clock** — 77.0 % vs 76.1 % of the FP32 peak (80.7 % vs 80.4 % in an
  earlier, cooler run) — with operands 4× smaller and the output half the size. On Pascal an exact
  E4M3 MAC costs one FFMA issue slot at best ([design.md](design.md)), so SGEMM speed is the ceiling.
- **Power, not code, sets the remaining gap.** The E4M3 × E4M3 kernel decodes both operands and draws
  up to ~210 W, hits the power limit and runs ~50 MHz below cuBLAS, so it takes 2 % longer (19.5 vs
  19.2 ms). With one operand in BF16 or FP32 it runs at the cuBLAS clock and finishes 3–4 % sooner.
- **GEMV is bandwidth-bound**: 0.237 ms for an 8192 × 8192 E4M3 matrix, 88 % of the DRAM peak (as fast
  as a device-to-device copy) and 4.4× cuBLAS SGEMV on FP32 weights; at n = 8 the kernel becomes
  compute- and register-bound and is still 3.1–3.9× faster than FP32 weights.
- **AdamW** works on 8 elements per thread with 16-byte loads and stores. With BF16 state it runs at
  282 GB/s, 88 % of the DRAM peak (as fast as a device-to-device copy): 3.3 ms for 64 M parameters, 2.2×
  faster than FP32 state. With FP32 state it moves 250 GB/s (78 %; median of a wide run-to-run spread,
  6.8–9 ms, whose fastest runs reach 87 %).
- **AdamW vs `torch.optim.AdamW`** (`bench/bench_torch_adamw.py`, same 64 M parameters, as one tensor or
  as 200 tensors of 320 K): against torch's fastest variant (`fused=True`) e4m3effma is 1.18–1.25×
  faster with FP32 parameters and state, and 2.44–2.69× faster with BF16 parameters and state — torch's
  fused kernel barely gains from BF16 (9.0 vs 9.3 ms), this one halves the time. With many tensors torch
  batches launches while e4m3effma launches one kernel per parameter, and it is still ahead.
