# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain100_ccpvdz_kfac_batch4096_iter10000_2gpu_fullnode_timebox20m`
Created: `2026-05-24T07:33:35.797647+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128331`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: n/a
- Seconds per pretrain row: n/a

## Config

- Pretrain method: `None`
- Target backend: `None`
- Basis: `None`
- Image cutoff: `None`
- Batch size: `None`
- Iterations: `None`
- Target chunk size: `None`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 100
- First/last step: 0 / 99
- First loss: 2.13493
- Last loss: 0.0184394
- Minimum loss: 0.0184394 at step 99
- Loss delta: -2.11649
- Loss drop fraction: 0.991363
- First-10 loss mean: 0.580741
- Last-10 loss mean: 0.019079
- Mean pmove: 0.841301
- Pmove range: [0.818359, 0.868896]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 99
- Mean steady step seconds: 0.143782
- Median steady step seconds: 0.080769
- P90 steady step seconds: 0.082089
- Mean target eval seconds: 0.057159
- Median target eval seconds: 0.056932
- P90 target eval seconds: 0.058059
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.085812
- Median JAX update seconds: 0.023054
- P90 JAX update seconds: 0.023492
- Target eval fraction: 0.397542
- Target transfer fraction: 0.000000
- JAX update fraction: 0.596824
- Step-0 seconds: 14.136643

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
