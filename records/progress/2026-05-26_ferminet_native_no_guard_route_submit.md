# FermiNet Native No-Guard Route Submit

Date: 2026-05-26

Task `0092` was cancelled because the heavyweight per-step NaN checks reduced
the native route speed by roughly `3--4x`.  Task `0093` restores the
production-speed route by disabling per-step NaN/debug scans.

Validated before submission:

- Build-only adapter check passed.
- Build summary confirms `check_nan=False` and `reset_if_nan=False`.
- `git diff --check` passed.
- `git -C external/ferminet diff --check` passed.

Task root:

```text
tasks/excited_state_nesvmc/0093_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard
```

Submission:

- Job ID: `130634`
- Job name: `0093_native_no_guard_4gpu40g`
- Partition/node: `amdgpu40g` / `gpu007`
- Resources: `gpu:4`, `64` CPU cores, exclusive node allocation
- Walltime: `04:00:00`
- Initial checked state: `RUNNING`
- Plan:
  `tasks/excited_state_nesvmc/0093_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard/outputs/slurm_plans/plan_amdgpu40g_no_guard.json`
- Slurm logs:
  `tasks/excited_state_nesvmc/0093_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard/logs/slurm/0093_native_no_guard_4gpu40g_130634.log`
  and
  `tasks/excited_state_nesvmc/0093_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard/logs/slurm/0093_native_no_guard_4gpu40g_130634.err`

Completion:

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
- Summary:
  `tasks/excited_state_nesvmc/0093_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000_fast_nan_guard/results/validation/native_ferminet_excited_summary.md`

Method decision:

- Continue future excited-state production work with the native simultaneous
  `vmc_overlap` route used by task `0093`.
- Keep fixed-ground overlap methods as experimental only.  Current issues are
  summarized in
  `docs/04_reports/excited_state_method_status_2026-05-26.md`.
