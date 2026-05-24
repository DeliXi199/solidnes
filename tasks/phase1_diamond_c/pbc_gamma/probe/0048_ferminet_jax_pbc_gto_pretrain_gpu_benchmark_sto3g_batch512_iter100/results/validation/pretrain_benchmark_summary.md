# FermiNet PBC-HF Pretrain Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_gpu_benchmark_sto3g_batch512_iter100`
Created: `2026-05-23T17:29:27.948805+00:00`
Row source: `pretrain_stats_csv`

## Runtime

- Job ID: `128116`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 130.000000
- Seconds per pretrain row: 1.300000

## Config

- Pretrain method: `pbc_hf`
- Target backend: `jax_pbc_gto`
- Basis: `sto-3g`
- Image cutoff: `2`
- Batch size: `512`
- Iterations: `100`
- Target chunk size: `0`
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
- Mean steady step seconds: 0.008500
- Median steady step seconds: 0.008494
- P90 steady step seconds: 0.009039
- Mean target eval seconds: 0.002028
- Median target eval seconds: 0.001986
- P90 target eval seconds: 0.002267
- Mean target transfer seconds: 0.000000
- Mean JAX update seconds: 0.006262
- Median JAX update seconds: 0.006198
- P90 JAX update seconds: 0.006618
- Target eval fraction: 0.238560
- Target transfer fraction: 0.000000
- JAX update fraction: 0.736712
- Step-0 seconds: 19.187471

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
