# FermiNet Benchmark Summary

Experiment: `diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_batch4096_iter50`
Created: `2026-05-25T06:45:32.178029+00:00`

## Runtime

- Job ID: `129249`
- Partition: `amdgpu40g`
- Node: `gpu005`
- GPU model: `a100_40gb`
- GRES: `gpu:4`
- Elapsed seconds: 149.000000
- Seconds per stats row: 2.980000
- Seconds per optimization step: 2.980000

## Config

- Network: `ferminet`
- Determinants: `8`
- Batch size: `4096`
- Optimizer: `kfac`
- Iterations: `50`
- Laplacian: `folx`
- Forward Laplacian enabled: `True`
- Precision profile: `speed`
- X64 enabled: `False`

## Stats

- Rows: 50
- First/last step: 0 / 49
- First energy: -22.054876 Ha
- Last energy: -24.691084 Ha
- Minimum energy: -24.748028 Ha at step 47
- Energy delta: -2.636208 Ha
- First-10 mean: -22.196684 Ha
- Last-10 mean: -24.432545 Ha
- First-50 mean: -23.150941 Ha
- Last-50 mean: -23.150941 Ha
- Mean pmove: 0.910310
- Pmove range: [0.908142, 0.914307]
- Last ewvar: 0.390211
- Finite checks: `True`

## Tail Estimate

- Tail rows: 25
- Tail start step: 25
- Tail energy mean: -23.863735 Ha
- Tail energy stderr: 0.111698 Ha
- Block energy stderr: 0.265737 Ha
- Tail pmove mean: 0.909669

## Diagnostics

- FOLX tile warnings: 0
- Traceback count: 0
