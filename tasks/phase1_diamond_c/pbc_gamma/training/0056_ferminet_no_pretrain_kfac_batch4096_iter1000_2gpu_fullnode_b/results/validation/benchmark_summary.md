# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_no_pretrain_kfac_batch4096_iter1000_2gpu_fullnode_b`
Created: `2026-05-24T05:38:02.852055+00:00`

## Runtime

- Job ID: `128323`
- Partition: `n/a`
- Node: `n/a`
- GPU model: `n/a`
- GRES: `n/a`
- Elapsed seconds: 1047.000000
- Seconds per stats row: 1.047000
- Seconds per optimization step: 1.047000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `1000`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `fp64`
- X64 enabled: `True`

## Stats

- Rows: 1000
- First/last step: 0 / 999
- First energy: -19.916310 Ha
- Last energy: -75.266479 Ha
- Minimum energy: -75.371321 Ha at step 907
- Energy delta: -55.350168 Ha
- First-10 mean: -20.081186 Ha
- Last-10 mean: -75.208688 Ha
- First-50 mean: -23.653705 Ha
- Last-50 mean: -75.172448 Ha
- Mean pmove: 0.712556
- Pmove range: [0.566248, 0.920313]
- Last ewvar: 0.002063
- Finite checks: `True`

## Tail Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -74.917542 Ha
- Tail energy stderr: 0.007209 Ha
- Block energy stderr: 0.074244 Ha
- Tail pmove mean: 0.636679

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
