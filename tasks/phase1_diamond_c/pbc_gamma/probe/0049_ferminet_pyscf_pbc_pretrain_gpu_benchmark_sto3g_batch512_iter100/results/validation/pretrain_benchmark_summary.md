# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_pyscf_pbc_pretrain_gpu_benchmark_sto3g_batch512_iter100`
Created: `2026-05-23T17:34:46.028985+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128120`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 133.000000
- Seconds per pretrain row: 1.330000

## Config

- Pretrain method: `pbc_hf`
- Target backend: `pyscf_pbc`
- Basis: `sto-3g`
- Image cutoff: `None`
- Batch size: `512`
- Iterations: `100`
- Target chunk size: `8192`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 100
- First/last step: 0 / 99
- First loss: 2.13895
- Last loss: 0.0187578
- Minimum loss: 0.018686 at step 98
- Loss delta: -2.12019
- Loss drop fraction: 0.991230
- First-10 loss mean: 0.583318
- Last-10 loss mean: 0.019176
- Mean pmove: 0.841660
- Pmove range: [0.792969, 0.884766]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 10
- Steady rows: 90
- Mean steady step seconds: 0.045811
- Median steady step seconds: 0.045846
- P90 steady step seconds: 0.046647
- Mean target eval seconds: 0.038902
- Median target eval seconds: 0.039070
- P90 target eval seconds: 0.039765
- Mean target transfer seconds: 0.000505
- Mean JAX update seconds: 0.006296
- Median JAX update seconds: 0.006275
- P90 JAX update seconds: 0.006482
- Target eval fraction: 0.849188
- Target transfer fraction: 0.011015
- JAX update fraction: 0.137429
- Step-0 seconds: 17.610693

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
