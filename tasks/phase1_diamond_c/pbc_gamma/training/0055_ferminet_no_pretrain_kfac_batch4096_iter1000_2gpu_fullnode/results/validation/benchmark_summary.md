# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_no_pretrain_kfac_batch4096_iter1000_2gpu_fullnode`
Created: `2026-05-24T05:37:43.140760+00:00`

## Runtime

- Job ID: `128322`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1035.000000
- Seconds per stats row: 1.035000
- Seconds per optimization step: 1.035000

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
- First energy: -20.013682 Ha
- Last energy: -75.106322 Ha
- Minimum energy: -75.298233 Ha at step 990
- Energy delta: -55.092640 Ha
- First-10 mean: -20.117638 Ha
- Last-10 mean: -75.189719 Ha
- First-50 mean: -23.454759 Ha
- Last-50 mean: -75.167610 Ha
- Mean pmove: 0.712574
- Pmove range: [0.562646, 0.919495]
- Last ewvar: 0.002152
- Finite checks: `True`

## Tail Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -74.908124 Ha
- Tail energy stderr: 0.007546 Ha
- Block energy stderr: 0.078206 Ha
- Tail pmove mean: 0.636694

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
