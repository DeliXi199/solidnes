# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_ccpvdz_kfac_batch4096_iter1000_2gpu_fullnode_b`
Created: `2026-05-24T04:19:02.497902+00:00`

## Runtime

- Job ID: `128316`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1153.000000
- Seconds per stats row: 1.153000
- Seconds per optimization step: 1.153000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `1000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `fp64`
- X64 enabled: `True`

## Stats

- Rows: 1000
- First/last step: 0 / 999
- First energy: -34.311641 Ha
- Last energy: -75.266107 Ha
- Minimum energy: -75.361109 Ha at step 870
- Energy delta: -40.954467 Ha
- First-10 mean: -35.139230 Ha
- Last-10 mean: -75.275132 Ha
- First-50 mean: -40.741058 Ha
- Last-50 mean: -75.250662 Ha
- Mean pmove: 0.710147
- Pmove range: [0.566016, 0.894690]
- Last ewvar: 0.001202
- Finite checks: `True`

## Tail Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -75.158029 Ha
- Tail energy stderr: 0.004283 Ha
- Block energy stderr: 0.038418 Ha
- Tail pmove mean: 0.636692

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
