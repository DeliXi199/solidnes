# 0091 FermiNet Native VMC Overlap Route Check

## Purpose

Rerun the original native FermiNet two-state excited-state route as a 10000-step
sanity check after the fixed-ground experimental branch showed issues.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_iter10000.yaml`
- Objective: `vmc_overlap`
- States: `2`
- Iterations: `10000`
- Batch size: `4096`
- Spin penalty: `0.0`
- `S^2` observable: disabled

## Requested Resources

- Partition: `amdgpu40g`
- Node count: `1`
- GPUs: `4`
- CPU cores: `64`
- Allocation: exclusive single node

## Submission

- Job ID: `130365`
- Job name: `0091_native_route_4gpu40g`
- Final state: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:20:33`
- Slurm scheduled node at submission time: `gpu007`
- Walltime: `04:00:00`
- Stdout:
  `tasks/excited_state_nesvmc/0091_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check/logs/slurm/0091_native_route_4gpu40g_130365.log`
- Stderr:
  `tasks/excited_state_nesvmc/0091_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check/logs/slurm/0091_native_route_4gpu40g_130365.err`

## Result

- Rows written: `10000`
- Last finite step: `3814`
- First non-finite step: `3815`
- Summary:
  `tasks/excited_state_nesvmc/0091_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check/results/validation/native_route_check_0091_summary.md`
