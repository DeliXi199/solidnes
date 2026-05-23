# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter10000_paper`
Created: `2026-05-23T07:10:36.175641+00:00`

## Runtime

- Job ID: `127848`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1748.000000
- Seconds per stats row: 0.174800
- Seconds per optimization step: 0.174800

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `10000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 10000
- First/last step: 0 / 9999
- First energy: -18.707180 Ha
- Last energy: -75.281360 Ha
- Minimum energy: -78.686110 Ha at step 511
- Energy delta: -56.574180 Ha
- First-10 mean: -18.740823 Ha
- Last-10 mean: -75.265029 Ha
- First-50 mean: -21.526318 Ha
- Last-50 mean: -75.260117 Ha
- Mean pmove: 0.553887
- Pmove range: [0.525635, 0.918372]
- Last ewvar: 0.001748
- Finite checks: `True`

## Tail Estimate

- Tail rows: 5000
- Tail start step: 5000
- Tail energy mean: -75.259508 Ha
- Tail energy stderr: 0.000899 Ha
- Block energy stderr: 0.002796 Ha
- Tail pmove mean: 0.535844

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
