# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000`
Created: `2026-05-23T12:02:35.682302+00:00`

## Runtime

- Job ID: `127992`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1325.000000
- Seconds per stats row: 0.662500
- Seconds per optimization step: 0.662500

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `none`
- Iterations: `2000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `fp64`
- X64 enabled: `True`

## Stats

- Rows: 2000
- First/last step: 0 / 1999
- First energy: -75.391116 Ha
- Last energy: -75.437477 Ha
- Minimum energy: -75.480005 Ha at step 77
- Energy delta: -0.046361 Ha
- First-10 mean: -75.409930 Ha
- Last-10 mean: -75.421399 Ha
- First-50 mean: -75.413363 Ha
- Last-50 mean: -75.415497 Ha
- Mean pmove: 0.536235
- Pmove range: [0.526404, 0.543787]
- Last ewvar: 0.000187
- Finite checks: `True`

## Tail Estimate

- Tail rows: 1000
- Tail start step: 1000
- Tail energy mean: -75.411863 Ha
- Tail energy stderr: 0.000386 Ha
- Block energy stderr: 0.000261 Ha
- Tail pmove mean: 0.536248

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
