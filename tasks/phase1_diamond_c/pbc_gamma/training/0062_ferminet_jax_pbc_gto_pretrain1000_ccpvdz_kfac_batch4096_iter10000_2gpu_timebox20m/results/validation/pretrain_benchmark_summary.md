# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain1000_ccpvdz_kfac_batch4096_iter10000_2gpu_fullnode_timebox20m`
Created: `2026-05-24T07:33:35.820363+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128334`
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

- Rows: 1000
- First/last step: 0 / 999
- First loss: 2.13491
- Last loss: 0.00645121
- Minimum loss: 0.00611499 at step 886
- Loss delta: -2.12846
- Loss drop fraction: 0.996978
- First-10 loss mean: 0.580722
- Last-10 loss mean: 0.006430
- Mean pmove: 0.877267
- Pmove range: [0.819092, 0.904053]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 999
- Mean steady step seconds: 0.089073
- Median steady step seconds: 0.082810
- P90 steady step seconds: 0.083619
- Mean target eval seconds: 0.058644
- Median target eval seconds: 0.058628
- P90 target eval seconds: 0.059342
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.029591
- Median JAX update seconds: 0.023369
- P90 JAX update seconds: 0.023582
- Target eval fraction: 0.658387
- Target transfer fraction: 0.000000
- JAX update fraction: 0.332212
- Step-0 seconds: 14.322693

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
