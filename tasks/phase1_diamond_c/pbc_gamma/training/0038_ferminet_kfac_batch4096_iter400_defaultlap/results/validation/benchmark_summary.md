# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_defaultlap`
Created: `2026-05-23T05:21:30.556267+00:00`

## Runtime

- Job ID: `127842`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:1`
- Elapsed seconds: 189.000000
- Seconds per stats row: 0.472500
- Seconds per optimization step: 0.472500

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `400`
- Laplacian: `default`
- Forward Laplacian enabled: `False`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 400
- First/last step: 0 / 399
- First energy: -20.792610 Ha
- Last energy: -73.746570 Ha
- Minimum energy: -73.897995 Ha at step 387
- Energy delta: -52.953960 Ha
- First-10 mean: -20.940738 Ha
- Last-10 mean: -73.594327 Ha
- First-50 mean: -22.187506 Ha
- Last-50 mean: -73.443896 Ha
- Mean pmove: 0.777777
- Pmove range: [0.637866, 0.910449]
- Last ewvar: 0.018472
- Finite checks: `True`

## Tail Estimate

- Tail rows: 200
- Tail start step: 200
- Tail energy mean: -71.539898 Ha
- Tail energy stderr: 0.151730 Ha
- Block energy stderr: 1.009560 Ha
- Tail pmove mean: 0.691532

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
