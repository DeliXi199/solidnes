# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_folx`
Created: `2026-05-23T05:52:37.572919+00:00`

## Runtime

- Job ID: `127845`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 394.000000
- Seconds per stats row: 0.197000
- Seconds per optimization step: 0.197000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `2000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 2000
- First/last step: 0 / 1999
- First energy: -20.790520 Ha
- Last energy: -75.177060 Ha
- Minimum energy: -78.169180 Ha at step 1706
- Energy delta: -54.386540 Ha
- First-10 mean: -20.927389 Ha
- Last-10 mean: -75.127121 Ha
- First-50 mean: -22.140761 Ha
- Last-50 mean: -75.111432 Ha
- Mean pmove: 0.587727
- Pmove range: [0.526416, 0.911157]
- Last ewvar: 0.002363
- Finite checks: `True`

## Tail Estimate

- Tail rows: 1000
- Tail start step: 1000
- Tail energy mean: -75.032381 Ha
- Tail energy stderr: 0.007936 Ha
- Block energy stderr: 0.055243 Ha
- Tail pmove mean: 0.536729

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
