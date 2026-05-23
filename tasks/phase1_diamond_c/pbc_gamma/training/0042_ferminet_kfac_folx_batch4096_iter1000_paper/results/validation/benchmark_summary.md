# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter1000_paper`
Created: `2026-05-23T06:31:44.775708+00:00`

## Runtime

- Job ID: `127847`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 428.000000
- Seconds per stats row: 0.428000
- Seconds per optimization step: 0.428000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `1000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 1000
- First/last step: 0 / 999
- First energy: -18.749450 Ha
- Last energy: -75.029800 Ha
- Minimum energy: -77.916960 Ha at step 914
- Energy delta: -56.280350 Ha
- First-10 mean: -18.750776 Ha
- Last-10 mean: -75.059691 Ha
- First-50 mean: -21.375507 Ha
- Last-50 mean: -75.071278 Ha
- Mean pmove: 0.713771
- Pmove range: [0.564502, 0.919287]
- Last ewvar: 0.006154
- Finite checks: `True`

## Tail Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -74.874834 Ha
- Tail energy stderr: 0.011366 Ha
- Block energy stderr: 0.077067 Ha
- Tail pmove mean: 0.637181

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
