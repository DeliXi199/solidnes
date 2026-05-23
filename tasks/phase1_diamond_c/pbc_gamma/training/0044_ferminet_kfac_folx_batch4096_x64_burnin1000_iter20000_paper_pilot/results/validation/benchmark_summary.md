# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot`
Created: `2026-05-23T11:17:18.069858+00:00`

## Runtime

- Job ID: `127898`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 8579.000000
- Seconds per stats row: 0.428950
- Seconds per optimization step: 0.428950

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `20000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `fp64`
- X64 enabled: `True`

## Stats

- Rows: 20000
- First/last step: 0 / 19999
- First energy: -16.696432 Ha
- Last energy: -75.400779 Ha
- Minimum energy: -75.570822 Ha at step 4349
- Energy delta: -58.704347 Ha
- First-10 mean: -17.017134 Ha
- Last-10 mean: -75.418459 Ha
- First-50 mean: -21.436065 Ha
- Last-50 mean: -75.416149 Ha
- Mean pmove: 0.544619
- Pmove range: [0.525708, 0.921448]
- Last ewvar: 0.000114
- Finite checks: `True`

## Tail Estimate

- Tail rows: 10000
- Tail start step: 10000
- Tail energy mean: -75.411406 Ha
- Tail energy stderr: 0.000138 Ha
- Block energy stderr: 0.001033 Ha
- Tail pmove mean: 0.535494

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
