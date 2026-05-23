# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100`
Created: `2026-05-23T04:47:52.380534+00:00`

## Runtime

- Job ID: `127830`
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
- First energy: -21.477194 Ha
- Last energy: -23.146812 Ha
- Minimum energy: -28.307531 Ha at step 68
- Energy delta: -1.669618 Ha
- First-10 mean: -20.914188 Ha
- Last-10 mean: -24.188218 Ha
- First-50 mean: -23.314920 Ha
- Last-50 mean: -25.320289 Ha
- Mean pmove: 0.901219
- Pmove range: [0.875000, 0.928125]
- Last ewvar: 0.796260
- Finite checks: `True`

## Tail Estimate

- Tail rows: 50
- Tail start step: 50
- Tail energy mean: -25.320289 Ha
- Tail energy stderr: 0.133177 Ha
- Block energy stderr: 0.364708 Ha
- Tail pmove mean: 0.899688

## Diagnostics

- FOLX tile warnings: 18
- Traceback count: 0
