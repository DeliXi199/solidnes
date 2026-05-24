# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_pbc_hf_pretrain_gpu_benchmark_batch512_iter100`
Created: `2026-05-24T01:47:27.444155+00:00`
Row source: `stderr_log`

## Runtime

- Job ID: `128113`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 147.000000
- Seconds per pretrain row: 13.363636

## Config

- Pretrain method: `pbc_hf`
- Target backend: `None`
- Basis: `ccpvdz`
- Image cutoff: `None`
- Batch size: `512`
- Iterations: `100`
- Target chunk size: `8192`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 11
- First/last step: 0 / 99
- First loss: 2.14205
- Last loss: 0.0179985
- Minimum loss: 0.0179985 at step 99
- Loss delta: -2.12405
- Loss drop fraction: 0.991598
- First-10 loss mean: 0.257810
- Last-10 loss mean: 0.045404
- Mean pmove: 0.844638
- Pmove range: [0.822266, 0.867188]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 10
- Mean steady step seconds: 0.058384
- Median steady step seconds: 0.058558
- P90 steady step seconds: 0.058849
- Mean target eval seconds: 0.051142
- Median target eval seconds: 0.051263
- P90 target eval seconds: 0.051536
- Mean target transfer seconds: 0.000634
- Mean JAX update seconds: 0.006495
- Median JAX update seconds: 0.006508
- P90 JAX update seconds: 0.006722
- Target eval fraction: 0.875963
- Target transfer fraction: 0.010857
- JAX update fraction: 0.111241
- Step-0 seconds: 18.441973

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
