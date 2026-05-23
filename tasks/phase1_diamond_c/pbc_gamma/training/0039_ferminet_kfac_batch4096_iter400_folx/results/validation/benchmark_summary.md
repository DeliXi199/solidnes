# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_folx`
Created: `2026-05-23T05:30:36.731971+00:00`

## Runtime

- Job ID: `127843`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 198.000000
- Seconds per stats row: 0.495000
- Seconds per optimization step: 0.495000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `400`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 400
- First/last step: 0 / 399
- First energy: -20.790490 Ha
- Last energy: -73.700320 Ha
- Minimum energy: -76.323910 Ha at step 362
- Energy delta: -52.909830 Ha
- First-10 mean: -20.921247 Ha
- Last-10 mean: -73.674769 Ha
- First-50 mean: -22.246368 Ha
- Last-50 mean: -73.488867 Ha
- Mean pmove: 0.777107
- Pmove range: [0.635010, 0.910693]
- Last ewvar: 0.054945
- Finite checks: `True`

## Tail Estimate

- Tail rows: 200
- Tail start step: 200
- Tail energy mean: -71.636613 Ha
- Tail energy stderr: 0.146046 Ha
- Block energy stderr: 0.974943 Ha
- Tail pmove mean: 0.690309

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
