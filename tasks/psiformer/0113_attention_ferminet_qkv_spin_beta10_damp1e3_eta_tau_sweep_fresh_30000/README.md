# 0113 FermiNet-QKV Eta/Tau Fresh 30000 Sweep

This task runs six fresh 30000-step two-state PsiFormer excited-state jobs.
All variants use the same FermiNet-QKV technical route and the same deterministic
initialization; only `eta0` and `tau` in the learning-rate schedule are changed.

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
- Blocked partitions include disabled `h20` through the submitter defaults

## Checkpoint Semantics

- Normal completion at step 29999 is expected to write
  `qmcjax_ckpt_029999.npz`.
- The FermiNet runner wraps training with `enforce_ferminet_final_checkpoint`,
  so a completed job is checked for the final checkpoint before exiting cleanly.
- Periodic checkpoints are saved every 10 minutes, so a 24-hour timeout before
  step 29999 should still leave a recent restart point.

## Variants

| Label | eta0 | tau | Experiment config | Slurm name | Job ID | Status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `eta1e2_tau15000` | `0.01` | `15000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e2_tau15000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta1e2-t15000-fresh30k` | 139207 | R on `gpuh2001` |
| `eta1e2_tau20000` | `0.01` | `20000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e2_tau20000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta1e2-t20000-fresh30k` | 139214 | R on `gpu004` |
| `eta1e2_tau10000` | `0.01` | `10000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta1e2_tau10000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta1e2-t10000-fresh30k` | 139209 | R on `gpu006` |
| `eta2e2_tau15000` | `0.02` | `15000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta2e2_tau15000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta2e2-t15000-fresh30k` | 139215 | PD on `h200,amdgpu80g,amdgpu40g` |
| `eta2e2_tau20000` | `0.02` | `20000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta2e2_tau20000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta2e2-t20000-fresh30k` | 139216 | PD on `h200,amdgpu80g,amdgpu40g` |
| `eta2e2_tau10000` | `0.02` | `10000` | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_eta2e2_tau10000_merge_none_batch4096_iter30000_fresh0113.yaml` | `solidnes-0113-ferm-b10-d1e3-eta2e2-t10000-fresh30k` | 139217 | PD on `h200,amdgpu80g,amdgpu40g` |

## Submission Notes

- Build-only verification passed for all six configs before submission.
- Slurm plans are written under `outputs/slurm_plans/`.
- Logs are written under `logs/slurm/`.

## Resubmission Note

The original pending jobs `139208`, `139210`, `139211`, and `139212` were
cancelled before running because the planner had pinned them with `--nodelist`.
They were resubmitted as `139214`, `139215`, `139216`, and `139217` with
`SOLIDNES_GPU_QUEUE_MODE=flexible`, so the `sbatch` commands specify only
`--partition h200,amdgpu80g,amdgpu40g` and do not include `--nodelist`.
