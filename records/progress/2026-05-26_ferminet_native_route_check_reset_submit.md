# FermiNet Native Route Check Reset Submit

Date: 2026-05-26

Task `0092` reruns the original native `vmc_overlap` route after fixing the
KFAC non-finite update guard.

Changes before submission:

- Added post-KFAC finite checking for updated params and optimizer state.
- Exposed `reset_if_nan` in the SolidNES FermiNet adapter.
- Enabled `check_nan` and `reset_if_nan` in the new 10000-step route-check
  train config.

Task root:

```text
tasks/excited_state_nesvmc/0092_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_reset_route_check
```

Validation before submission:

- `compileall` passed for `external/ferminet/ferminet/train.py` and
  `src/solidnes/backends/ferminet_adapter.py`.
- Native overlap validation passed:
  `scripts/validation/check_ferminet_native_overlap_loss_alignment.py`.
- Build-only adapter check passed and confirmed `check_nan=True` and
  `reset_if_nan=True`.
- `git diff --check` and `git -C external/ferminet diff --check` passed.

Submission:

- Job ID: `130513`
- Job name: `0092_native_reset_route_4gpu40g`
- Partition/node: `amdgpu40g` / `gpu007`
- Resources: `gpu:4`, `64` CPU cores, exclusive node allocation
- Walltime: `04:00:00`
- Initial checked state: `RUNNING`
- Plan:
  `tasks/excited_state_nesvmc/0092_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_reset_route_check/outputs/slurm_plans/plan_amdgpu40g_reset.json`
- Slurm logs:
  `tasks/excited_state_nesvmc/0092_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_reset_route_check/logs/slurm/0092_native_reset_route_4gpu40g_130513.log`
  and
  `tasks/excited_state_nesvmc/0092_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_reset_route_check/logs/slurm/0092_native_reset_route_4gpu40g_130513.err`
