# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_ccpvdz_kfac_batch4096_iter1000`
Created: `2026-05-24T03:16:37.635835+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128298`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: n/a
- Seconds per pretrain row: n/a

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
- First loss: 2.1349
- Last loss: 0.00585814
- Minimum loss: 0.00571214 at step 977
- Loss delta: -2.12905
- Loss drop fraction: 0.997256
- First-10 loss mean: 0.580563
- Last-10 loss mean: 0.005810
- Mean pmove: 0.877025
- Pmove range: [0.820068, 0.904053]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 10
- Steady rows: 990
- Mean steady step seconds: 0.150885
- Median steady step seconds: 0.150848
- P90 steady step seconds: 0.151658
- Mean target eval seconds: 0.109120
- Median target eval seconds: 0.109065
- P90 target eval seconds: 0.109855
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.041302
- Median JAX update seconds: 0.041279
- P90 JAX update seconds: 0.041610
- Target eval fraction: 0.723202
- Target transfer fraction: 0.000000
- JAX update fraction: 0.273729
- Step-0 seconds: 18.722209

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
