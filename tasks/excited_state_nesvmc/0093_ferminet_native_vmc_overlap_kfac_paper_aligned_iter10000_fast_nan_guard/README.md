# 0093 FermiNet Native VMC Overlap No-Guard Route

## Purpose

Replace task `0092`, whose per-step NaN debug checks made the original native
FermiNet two-state route several times slower than expected.

This run restores the production-speed route:

- `check_nan: false`
- `reset_if_nan: false`
- `spin_penalty: 0.0`
- `S^2` observable disabled

NaN monitoring should be done out of process from logs and result files rather
than by scanning the full parameter or optimizer trees inside every training
step.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_iter10000_fast_nan_guard.yaml`
- Objective: `vmc_overlap`
- States: `2`
- Iterations: `10000`
- Batch size: `4096`

## Submission

- Job ID: `130634`
- Job name: `0093_native_no_guard_4gpu40g`
- Partition/node: `amdgpu40g` / `gpu007`
- Resources: `gpu:4`, `64` CPU cores, exclusive node allocation
- Walltime: `04:00:00`
- State after submission check: `RUNNING`
- Slurm stdout:
  `logs/slurm/0093_native_no_guard_4gpu40g_130634.log`
- Slurm stderr:
  `logs/slurm/0093_native_no_guard_4gpu40g_130634.err`

## Result

- Slurm state: `COMPLETED`, exit `0:0`
- Elapsed: `00:20:36`
- Rows: `10000`
- Non-finite CSV rows: `0`
- Non-finite diagnostic frames: `0`
- Final scalar energy: `-74.981740 Ha`
- Tail-100 scalar energy mean: `-75.029858 Ha`
- Final state energies: `[-75.126656, -74.691872] Ha`
- Final gap: `11.831 eV`
- Tail-100 gap mean/median/std: `9.788 / 9.132 / 4.925 eV`
- Final symmetric overlap offdiag: `0.0`
- Final overlap penalty offdiag: `0.0`
