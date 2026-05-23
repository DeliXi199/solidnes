# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_folx`
Created: `2026-05-23T05:13:40.022514+00:00`

## Runtime

- Job ID: `127841`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 155.000000
- Seconds per stats row: 1.550000
- Seconds per optimization step: 1.550000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `adam`
- Iterations: `100`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 100
- First/last step: 0 / 99
- First energy: -21.042307 Ha
- Last energy: -29.373300 Ha
- Minimum energy: -29.373300 Ha at step 99
- Energy delta: -8.330993 Ha
- First-10 mean: -22.479195 Ha
- Last-10 mean: -29.213502 Ha
- First-50 mean: -26.018495 Ha
- Last-50 mean: -28.696910 Ha
- Mean pmove: 0.900885
- Pmove range: [0.882520, 0.910571]
- Last ewvar: 0.076433
- Finite checks: `True`

## Tail Estimate

- Tail rows: 50
- Tail start step: 50
- Tail energy mean: -28.696910 Ha
- Tail energy stderr: 0.059159 Ha
- Block energy stderr: 0.195951 Ha
- Tail pmove mean: 0.901553

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
