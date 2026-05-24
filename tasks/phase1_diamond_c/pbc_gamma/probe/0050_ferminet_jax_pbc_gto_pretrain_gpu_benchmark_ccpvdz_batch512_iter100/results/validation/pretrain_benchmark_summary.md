# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_gpu_benchmark_ccpvdz_batch512_iter100`
Created: `2026-05-24T01:46:34.738464+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128288`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 149.000000
- Seconds per pretrain row: 1.490000

## Config

- Pretrain method: `pbc_hf`
- Target backend: `jax_pbc_gto`
- Basis: `ccpvdz`
- Image cutoff: `3`
- Batch size: `512`
- Iterations: `100`
- Target chunk size: `0`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 100
- First/last step: 0 / 99
- First loss: 2.14452
- Last loss: 0.0178482
- Minimum loss: 0.0178482 at step 99
- Loss delta: -2.12667
- Loss drop fraction: 0.991677
- First-10 loss mean: 0.584670
- Last-10 loss mean: 0.018507
- Mean pmove: 0.841172
- Pmove range: [0.796875, 0.898438]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 10
- Steady rows: 90
- Mean steady step seconds: 0.023191
- Median steady step seconds: 0.023308
- P90 steady step seconds: 0.024137
- Mean target eval seconds: 0.015405
- Median target eval seconds: 0.015405
- P90 target eval seconds: 0.016098
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.007429
- Median JAX update seconds: 0.007469
- P90 JAX update seconds: 0.007823
- Target eval fraction: 0.664296
- Target transfer fraction: 0.000000
- JAX update fraction: 0.320363
- Step-0 seconds: 19.661485

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
