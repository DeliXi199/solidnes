# 0092 FermiNet Native VMC Overlap Reset Route Check

## Purpose

Rerun the original native FermiNet two-state excited-state route after fixing
the KFAC non-finite update guard exposed by task `0091`.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000_reset_route_check.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_iter10000_reset.yaml`
- Objective: `vmc_overlap`
- States: `2`
- Iterations: `10000`
- Batch size: `4096`
- Spin penalty: `0.0`
- `S^2` observable: disabled
- `check_nan`: enabled
- `reset_if_nan`: enabled

## Requested Resources

- Partition: `amdgpu40g`
- Node count: `1`
- GPUs: `4`
- CPU cores: `64`
- Allocation: exclusive single node

## Submission

- Job ID: `130513`
- Job name: `0092_native_reset_route_4gpu40g`
- Partition/node: `amdgpu40g` / `gpu007`
- Resources: `gpu:4`, `64` CPU cores, exclusive node allocation
- Walltime: `04:00:00`
- State after submission check: `RUNNING`
- Slurm stdout:
  `logs/slurm/0092_native_reset_route_4gpu40g_130513.log`
- Slurm stderr:
  `logs/slurm/0092_native_reset_route_4gpu40g_130513.err`
