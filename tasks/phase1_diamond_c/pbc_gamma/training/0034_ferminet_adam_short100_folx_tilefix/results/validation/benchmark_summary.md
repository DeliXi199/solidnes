# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100_folx_tilefix`
Created: `2026-05-23T04:53:48.898927+00:00`

## Runtime

- Job ID: `127833`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 117.000000
- Seconds per stats row: 1.170000
- Seconds per optimization step: 1.170000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `64`
- Optimizer: `adam`
- Iterations: `100`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 100
- First/last step: 0 / 99
- First energy: -19.545017 Ha
- Last energy: -23.293020 Ha
- Minimum energy: -27.982136 Ha at step 65
- Energy delta: -3.748003 Ha
- First-10 mean: -21.309142 Ha
- Last-10 mean: -24.472733 Ha
- First-50 mean: -22.971573 Ha
- Last-50 mean: -25.811617 Ha
- Mean pmove: 0.903578
- Pmove range: [0.870313, 0.934375]
- Last ewvar: 1.103648
- Finite checks: `True`

## Tail Estimate

- Tail rows: 50
- Tail start step: 50
- Tail energy mean: -25.811617 Ha
- Tail energy stderr: 0.142632 Ha
- Block energy stderr: 0.347975 Ha
- Tail pmove mean: 0.902938

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
