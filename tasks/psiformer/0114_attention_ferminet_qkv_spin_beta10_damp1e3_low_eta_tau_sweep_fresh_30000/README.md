# 0114 FermiNet-QKV Low-Eta/Tau Fresh 30000 Sweep

This task runs six fresh 30000-step two-state PsiFormer excited-state jobs.
It follows task 0113 exactly except that the base learning rates are lower:
`eta0=0.005` and `eta0=0.001`, each combined with `tau=10000`, `15000`, and
`20000`.

## Excited-State Method Summary

- Backend: native FermiNet training adapter
- Model: PsiFormer PBC paper x64 model with FermiNet-style separate Q/K/V attention
- Objective: `vmc_overlap` two-state excited-state optimization
- State parameters: independent per-state parameter trees, `merge_keys: []`
- Spin control: DeepQMC-style loss-level `beta * <S^2>`, `beta=10.0`, target singlet `S^2=0`
- Spin logging: `log_spin_by_state: true`, full S2 matrix observable disabled
- KFAC: `damping=0.001`, `norm_constraint=0.001`, `invert_every=1`, `cov_update_every=1`
- Schedule: `eta(t)=eta0/(1+t/tau)` with `learning_rate_decay=1.0`
- Initialization: fresh start with `deterministic: true`; upstream FermiNet uses fixed seed 23 in this mode
- No checkpoint restore is used in this task

## Fixed Runtime Settings

- Iterations: 30000
- Batch size: 4096
- Precision: x64 / fp64
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Slurm time limit: 24 hours
- Requested partitions: `h200,amdgpu80g,amdgpu40g`
- Queue mode: flexible partition queue, no pinned `--nodelist`

## Variants

| Label | eta0 | tau | Experiment config | Slurm name | Job ID | Status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `eta5e3_tau10000` | `0.005` | `10000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta5e3_tau10000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta5e3-t10000-fresh30k` | 139618 | COMPLETED `0:0` on `gpuh2001` |
| `eta5e3_tau15000` | `0.005` | `15000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta5e3_tau15000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta5e3-t15000-fresh30k` | 139657 | COMPLETED `0:0` on `gpu004`; original 139619 cancelled |
| `eta5e3_tau20000` | `0.005` | `20000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta5e3_tau20000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta5e3-t20000-fresh30k` | 139658 | COMPLETED `0:0` on `gpu002`; original 139620 cancelled |
| `eta1e3_tau10000` | `0.001` | `10000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e3_tau10000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta1e3-t10000-fresh30k` | 139659 | COMPLETED `0:0` on `gpuh2001`; original 139621 cancelled |
| `eta1e3_tau15000` | `0.001` | `15000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e3_tau15000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta1e3-t15000-fresh30k` | 139660 | COMPLETED `0:0` on `gpuh2001`; original 139622 cancelled |
| `eta1e3_tau20000` | `0.001` | `20000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e3_tau20000_merge_none_batch4096_iter30000_fresh0114.yaml` | `solidnes-0114-ferm-b10-d1e3-eta1e3-t20000-fresh30k` | 139661 | COMPLETED `0:0` on `gpuh2001`; original 139623 cancelled |

## Submission Notes

- Configs are copied from the completed 0113 sweep and changed only in
  `eta0`, task/output names, and related diagnostics.
- Build-only verification passed for all six configs with
  `.venv/ferminet-jax0101-cuda12/bin/python`.
- Slurm plans are written under `outputs/slurm_plans/`.
- Logs are written under `logs/slurm/`.

## Cancellation Note

Jobs 139619-139623 were cancelled while still pending, then resubmitted as
139657-139661. Jobs 139618 and 139657-139661 all completed with exit code
`0:0`.

## Analysis Outputs

- `analysis/0114_low_eta_tau_sweep_comparison.md`
- `analysis/0114_low_eta_tau_sweep_summary.csv`
- `analysis/0114_low_eta_tau_sweep_combined_timeseries.csv`
- `analysis/0114_low_eta_tau_sweep_last10000_comparison.md`
- `analysis/0114_low_eta_tau_sweep_last10000_summary.csv`
- `analysis/0114_low_eta_tau_sweep_last10000_combined_timeseries.csv`
- `analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_last10000_comparison.md`
- `analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_last10000_summary.csv`
- `analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_combined_timeseries.csv`
- `analysis/fixed_tau_eta_comparison/*_rolling_var_after20000_window1000.{png,svg}`
- `analysis/fixed_tau_eta_comparison/*_rolling_std_after20000_window1000.{png,svg}`
- `analysis/fixed_tau_eta_comparison/*_rolling_abs_delta_after20000_window1000.{png,svg}`
- `analysis/fixed_tau_eta_comparison/*_ewvar_rolling_after20000_window1000.{png,svg}`

## Default Parameter Decision

The combined 0113/0114 fixed-tau analysis selects `eta0=0.02`, `tau=10000`,
and `learning_rate_decay=1.0` as the default optimizer schedule for future
PsiFormer excited-state calculations unless a task explicitly declares a new
sweep or ablation.

Reference configs:

- `configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml`
- `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml`

Saved snapshot:

- `records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/default_parameter_decision.md`
- `records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/fixed_tau_eta_comparison_analysis_20260608.tar.gz`
