# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain500_ccpvdz_kfac_batch4096_iter10000_2gpu_fullnode_timebox20m`
Created: `2026-05-24T07:33:35.822940+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128333`
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

- Rows: 500
- First/last step: 0 / 499
- First loss: 2.13492
- Last loss: 0.00754142
- Minimum loss: 0.00739126 at step 468
- Loss delta: -2.12738
- Loss drop fraction: 0.996468
- First-10 loss mean: 0.580735
- Last-10 loss mean: 0.007532
- Mean pmove: 0.868936
- Pmove range: [0.819336, 0.900879]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 499
- Mean steady step seconds: 0.095087
- Median steady step seconds: 0.082707
- P90 steady step seconds: 0.083624
- Mean target eval seconds: 0.058388
- Median target eval seconds: 0.058532
- P90 target eval seconds: 0.059287
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.035860
- Median JAX update seconds: 0.023360
- P90 JAX update seconds: 0.023606
- Target eval fraction: 0.614050
- Target transfer fraction: 0.000000
- JAX update fraction: 0.377122
- Step-0 seconds: 14.098208

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
