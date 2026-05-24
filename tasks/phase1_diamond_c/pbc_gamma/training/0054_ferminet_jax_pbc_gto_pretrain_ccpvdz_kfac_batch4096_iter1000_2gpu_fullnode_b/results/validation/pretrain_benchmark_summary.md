# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_ccpvdz_kfac_batch4096_iter1000_2gpu_fullnode_b`
Created: `2026-05-24T04:18:58.750311+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128316`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1153.000000
- Seconds per pretrain row: 1.153000

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
- First loss: 2.14111
- Last loss: 0.00609598
- Minimum loss: 0.00609598 at step 999
- Loss delta: -2.13501
- Loss drop fraction: 0.997153
- First-10 loss mean: 0.584492
- Last-10 loss mean: 0.006244
- Mean pmove: 0.877482
- Pmove range: [0.817627, 0.903809]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 10
- Steady rows: 990
- Mean steady step seconds: 0.082987
- Median steady step seconds: 0.082910
- P90 steady step seconds: 0.083686
- Mean target eval seconds: 0.058686
- Median target eval seconds: 0.058690
- P90 target eval seconds: 0.059404
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.023468
- Median JAX update seconds: 0.023405
- P90 JAX update seconds: 0.023612
- Target eval fraction: 0.707166
- Target transfer fraction: 0.000000
- JAX update fraction: 0.282796
- Step-0 seconds: 13.509715

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
