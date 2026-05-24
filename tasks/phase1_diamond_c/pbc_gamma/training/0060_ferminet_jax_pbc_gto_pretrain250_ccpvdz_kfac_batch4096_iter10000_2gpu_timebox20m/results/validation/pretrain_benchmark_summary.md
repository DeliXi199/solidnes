# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain250_ccpvdz_kfac_batch4096_iter10000_2gpu_fullnode_timebox20m`
Created: `2026-05-24T07:33:35.815574+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128332`
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

- Rows: 250
- First/last step: 0 / 249
- First loss: 2.14111
- Last loss: 0.010401
- Minimum loss: 0.0103654 at step 248
- Loss delta: -2.13071
- Loss drop fraction: 0.995142
- First-10 loss mean: 0.584489
- Last-10 loss mean: 0.010573
- Mean pmove: 0.854943
- Pmove range: [0.817627, 0.884277]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 249
- Mean steady step seconds: 0.107756
- Median steady step seconds: 0.082458
- P90 steady step seconds: 0.083208
- Mean target eval seconds: 0.058316
- Median target eval seconds: 0.058387
- P90 target eval seconds: 0.059073
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.048608
- Median JAX update seconds: 0.023261
- P90 JAX update seconds: 0.023437
- Target eval fraction: 0.541189
- Target transfer fraction: 0.000000
- JAX update fraction: 0.451093
- Step-0 seconds: 14.091918

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
