# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_defaultlap`
Created: `2026-05-23T05:52:37.533558+00:00`

## Runtime

- Job ID: `127844`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 557.000000
- Seconds per stats row: 0.278500
- Seconds per optimization step: 0.278500

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `2000`
- Laplacian: `default`
- Forward Laplacian enabled: `False`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 2000
- First/last step: 0 / 1999
- First energy: -21.227180 Ha
- Last energy: -74.983570 Ha
- Minimum energy: -78.825424 Ha at step 1689
- Energy delta: -53.756390 Ha
- First-10 mean: -21.220309 Ha
- Last-10 mean: -75.028169 Ha
- First-50 mean: -22.621719 Ha
- Last-50 mean: -75.020628 Ha
- Mean pmove: 0.586968
- Pmove range: [0.525293, 0.911938]
- Last ewvar: 0.002558
- Finite checks: `True`

## Tail Estimate

- Tail rows: 1000
- Tail start step: 1000
- Tail energy mean: -74.931896 Ha
- Tail energy stderr: 0.007377 Ha
- Block energy stderr: 0.048679 Ha
- Tail pmove mean: 0.536195

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
