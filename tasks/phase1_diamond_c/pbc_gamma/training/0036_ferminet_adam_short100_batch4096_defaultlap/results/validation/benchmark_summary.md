# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_defaultlap`
Created: `2026-05-23T05:13:40.010437+00:00`

## Runtime

- Job ID: `127840`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 110.000000
- Seconds per stats row: 1.100000
- Seconds per optimization step: 1.100000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `adam`
- Iterations: `100`
- Laplacian: `default`
- Forward Laplacian enabled: `False`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 100
- First/last step: 0 / 99
- First energy: -21.071262 Ha
- Last energy: -29.266440 Ha
- Minimum energy: -29.283728 Ha at step 94
- Energy delta: -8.195178 Ha
- First-10 mean: -22.508654 Ha
- Last-10 mean: -29.158609 Ha
- First-50 mean: -25.958588 Ha
- Last-50 mean: -28.578944 Ha
- Mean pmove: 0.901304
- Pmove range: [0.882324, 0.911426]
- Last ewvar: 0.092456
- Finite checks: `True`

## Tail Estimate

- Tail rows: 50
- Tail start step: 50
- Tail energy mean: -28.578944 Ha
- Tail energy stderr: 0.055902 Ha
- Block energy stderr: 0.185070 Ha
- Tail pmove mean: 0.901456

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
