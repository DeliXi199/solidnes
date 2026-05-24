# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_jax_pbc_gto_pretrain_ccpvdz_kfac_batch4096_iter1000_2gpu_fullnode_a`
Created: `2026-05-24T04:19:02.514240+00:00`

## Runtime

- Job ID: `128315`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model: `a100_80gb`
- GRES: `gpu:2`
- Elapsed seconds: 1173.000000
- Seconds per stats row: 1.173000
- Seconds per optimization step: 1.173000

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
- First energy: -34.336595 Ha
- Last energy: -75.134508 Ha
- Minimum energy: -75.375215 Ha at step 991
- Energy delta: -40.797913 Ha
- First-10 mean: -34.953916 Ha
- Last-10 mean: -75.259876 Ha
- First-50 mean: -40.663122 Ha
- Last-50 mean: -75.254433 Ha
- Mean pmove: 0.709987
- Pmove range: [0.564880, 0.895020]
- Last ewvar: 0.004003
- Finite checks: `True`

## Tail Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -75.134893 Ha
- Tail energy stderr: 0.004880 Ha
- Block energy stderr: 0.044589 Ha
- Tail pmove mean: 0.636575

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
