# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100_defaultlap`
Created: `2026-05-23T05:04:09.752201+00:00`

## Runtime

- Job ID: `127837`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 57.000000
- Seconds per stats row: 0.570000
- Seconds per optimization step: 0.570000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `64`
- Optimizer: `adam`
- Iterations: `100`
- Laplacian: `default`
- Forward Laplacian enabled: `False`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 100
- First/last step: 0 / 99
- First energy: -21.763638 Ha
- Last energy: -25.978733 Ha
- Minimum energy: -37.044224 Ha at step 89
- Energy delta: -4.215095 Ha
- First-10 mean: -21.279807 Ha
- Last-10 mean: -25.503886 Ha
- First-50 mean: -23.996492 Ha
- Last-50 mean: -26.592552 Ha
- Mean pmove: 0.899469
- Pmove range: [0.871875, 0.926562]
- Last ewvar: 4.953837
- Finite checks: `True`

## Tail Estimate

- Tail rows: 50
- Tail start step: 50
- Tail energy mean: -26.592552 Ha
- Tail energy stderr: 0.239804 Ha
- Block energy stderr: 0.363803 Ha
- Tail pmove mean: 0.900719

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
