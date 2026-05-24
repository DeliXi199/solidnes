# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_pbc_hf_pretrain_benchmark_batch512_iter100`
Created: `2026-05-23T17:09:51.561461+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `n/a`
- Partition: `n/a`
- Node: `n/a`
- GPU model: `n/a`
- GRES: `n/a`
- Elapsed seconds: n/a
- Seconds per pretrain row: n/a

## Config

- Pretrain method: `pbc_hf`
- Basis: `ccpvdz`
- Batch size: `512`
- Iterations: `100`
- Target chunk size: `8192`
- Precision profile: `fp64`
- X64 enabled: `True`

## Pretrain

- Rows: 100
- First/last step: 0 / 99
- First loss: 2.14206
- Last loss: 0.0179841
- Minimum loss: 0.0179841 at step 99
- Loss delta: -2.12407
- Loss drop fraction: 0.991604
- First-10 loss mean: 0.583542
- Last-10 loss mean: 0.018543
- Mean pmove: 0.843477
- Pmove range: [0.812500, 0.886719]
- Finite checks: `True`

## Timing

- Warmup rows excluded: 1
- Steady rows: 99
- Mean steady step seconds: 1.437765
- Mean target eval seconds: 0.386994
- Mean target transfer seconds: 0.017376
- Mean JAX update seconds: 1.033282
- Target eval fraction: 0.269164
- Target transfer fraction: 0.012085
- JAX update fraction: 0.718672
- Step-0 seconds: 3.261245

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
