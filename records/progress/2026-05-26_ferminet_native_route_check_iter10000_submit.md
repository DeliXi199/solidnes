# 2026-05-26 FermiNet native excited-state route check submission

## Scope

- Task: `0091_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check`
- Experiment:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000_route_check.yaml`
- Objective: `vmc_overlap`
- States: `2`
- Iterations: `10000`
- Batch size: `4096`
- Spin penalty: `0.0`
- `S^2` observable: disabled

## Checks

- `run_ferminet_train.py --build-only` passed.
- `build_ferminet_config.py --create-output-dirs` passed.
- Build summary confirmed `fixed_ground_checkpoint: None` and
  `fixed_ground_symmetric_sampling: False`.

## Slurm

- Job ID: `130365`
- Partition: `amdgpu40g`
- Resources: one exclusive node, `gpu:4`, `64` CPU cores, `--mem=0`,
  walltime `04:00:00`.
- State after submission: `PD (Resources)`.
