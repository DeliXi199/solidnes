# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_ccpvdz_kfac_batch4096_iter1000_2gpu_fullnode_a`
Created: `2026-05-24T04:18:58.747222+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128315`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1173.000000
- Seconds per pretrain row: 1.173000

## Config

- Pretrain method: `pbc_hf`
- Target backend: `jax_pbc_gto`
- Basis: `ccpvdz`
- Image cutoff: `3`
- Batch size: `4096`
- Iterations: `1000`
- Target chunk size: `0`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 1000
- First/last step: 0 / 999
- First loss: 2.14112
- Last loss: 0.00622516
- Minimum loss: 0.00619445 at step 998
- Loss delta: -2.13489
- Loss drop fraction: 0.997093
- First-10 loss mean: 0.584499
- Last-10 loss mean: 0.006281
- Mean pmove: 0.877656
- Pmove range: [0.817383, 0.904541]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 10
- Steady rows: 990
- Mean steady step seconds: 0.083744
- Median steady step seconds: 0.083536
- P90 steady step seconds: 0.084406
- Mean target eval seconds: 0.059224
- Median target eval seconds: 0.059161
- P90 target eval seconds: 0.059852
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.023656
- Median JAX update seconds: 0.023499
- P90 JAX update seconds: 0.023761
- Target eval fraction: 0.707205
- Target transfer fraction: 0.000000
- JAX update fraction: 0.282475
- Step-0 seconds: 13.687438

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
