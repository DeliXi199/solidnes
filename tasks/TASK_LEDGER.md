# SolidNES Task Ledger

Last updated: 2026-06-08, Asia/Shanghai

This is now a compact index. Do not use it as the default Codex startup file.

The exact pre-split full ledger is archived at:

```text
records/archive/2026-06-01_context_split/TASK_LEDGER.md
```

Readable split ledgers:

```text
tasks/ledger/0001_0062_ground_pretrain.md
tasks/ledger/0063_0093_ferminet_excited.md
tasks/ledger/0094_0103_psiformer_deepqmc.md
tasks/ledger/0104_spin_penalty_alignment.md
tasks/ledger/0105_spin_state_specific_alignment.md
tasks/ledger/0106_attention_qkv_spin0.md
tasks/ledger/0107_attention_qkv_spin0_rerun.md
tasks/ledger/0108_attention_qkv_spin0_continue_20000.md
tasks/ledger/0109_attention_qkv_spin20_continue_10000.md
tasks/ledger/0110_attention_qkv_spin_beta10_damp2e3_continue_10000.md
tasks/ledger/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000.md
tasks/ledger/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000.md
tasks/ledger/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000.md
tasks/ledger/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000.md
```

## Current Position

```text
Next available task number: 0115
Current source-code excited-state mainline: fused-QKV no-merge DeepQMC-aligned
Default excited-state optimizer schedule: eta0=0.02, tau=10000, decay=1.0
Default validated PsiFormer excited-state reference: FermiNet-QKV, beta=10, damping=1e-3
Default merge_keys: []
Non-empty merge_keys: explicit comparison branches
Optional spin penalty: DeepQMC-style loss-level beta * <S^2>
Fixed-ground: explicit-only; do not select unless the user says fixed-ground
```

## Recent Tasks

| Task | Status | Root | Summary |
| --- | --- | --- | --- |
| 0094 | completed | `tasks/psiformer/0094_psiformer_attention_build_benchmark` | Added configurable PsiFormer attention and validated fused-QKV forward exactness. |
| 0095 | completed | `tasks/psiformer/0095_psiformer_native_training_smoke` | Validated native PsiFormer training path; full-training timing bottleneck was outside attention projection. |
| 0096 | completed | `tasks/psiformer/0096_psiformer_attention_full_stack` | Completed no-pretrain attention validation; superseded for method selection by 0097-0103. |
| 0097 | completed | `tasks/psiformer/0097_deepqmc_aligned_excited_state` | Completed 10000-step DeepQMC-aligned independent-state attention variants after KFAC shape fix. |
| 0098 | completed | `tasks/psiformer/0098_deepqmc_alignment_gpu_smoke` | Validated source-level DeepQMC alignment changes on Slurm GPU after donation-aliasing fix. |
| 0099 | completed | `tasks/psiformer/0099_deepqmc_alignment_speed_benchmark` | Compared small diagonal-on/off speed paths. |
| 0100 | completed | `tasks/psiformer/0100_deepqmc_alignment_4gpu_speed_benchmark` | Measured 4GPU diagonal fast-path speedup: about 1.71x. |
| 0101 | completed | `tasks/psiformer/0101_deepqmc_merge_keys_4gpu_sweep` | Ran 200-step merge-key sweep; merge remained a comparison branch question. |
| 0102 | completed | `tasks/psiformer/0102_deepqmc_nonmerge_alignment_final_validation` | Validated non-merge alignment and diagonal path speed at 4GPU scale. |
| 0103 | completed | `tasks/psiformer/0103_attention_merge_keys_4gpu_10000` | Completed 10000-step attention x merge-key comparison; supports no-merge source default. |
| 0104 | completed | `tasks/excited_state_nesvmc/0104_deepqmc_spin_penalty_alignment_gpu_smoke` | SLURM job 135738 validated DeepQMC-style loss-level spin penalty with beta=10 on the no-merge PsiFormer route; finite energy/overlap/S2 outputs. |
| 0105 | completed | `tasks/excited_state_nesvmc/0105_deepqmc_spin_state_specific_gpu_smoke` | SLURM job 135764 validated the final DeepQMC state-specific local S² estimator in the loss-level spin penalty; finite energy/overlap/S2 outputs. |
| 0106 | submitted | `tasks/psiformer/0106_attention_qkv_spin0_4gpu_10000` | Cancelled initial full-S2-matrix submissions 135878/135879 before start; submitted corrected jobs 135932 (`fused_qkv`) and 135931 (`ferminet`) for 10000-step no-merge PsiFormer attention QKV comparison with beta=10 spin penalty targeting S=0, `log_spin_by_state=true`, and `observables_s2=false`. |
| 0112 | completed | `tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000` | Completed jobs 138677-138682 and analyzed the learning-rate sweep continuing the two completed 0108 step-29999 attention routes for 10000 more steps with `damping=0.001`; swept only base learning rate over `0.02`, `0.01`, and `0.005`. |
| 0113 | completed | `tasks/psiformer/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000` | Completed six fresh 30000-step FermiNet-QKV excited-state jobs with deterministic shared initialization, `damping=0.001`, `beta=10`, and eta/tau sweep over `(0.01, 15000)`, `(0.01, 20000)`, `(0.01, 10000)`, `(0.02, 15000)`, `(0.02, 20000)`, `(0.02, 10000)`; combined with 0114 for fixed-tau eta analysis. |
| 0114 | completed | `tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000` | Completed jobs 139618 and 139657-139661 for the low-eta fresh sweep; generated E0/E1/gap/spin rolling means plus rolling std/variance/one-step-change plots. Combined 0113/0114 analysis selected `eta0=0.02, tau=10000` as the default excited-state optimizer schedule and saved a snapshot under `records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/`. |

## Maintenance Rules

- Keep this file short.
- Put detailed task history in `tasks/ledger/` and task-local README files.
- Keep exact historical snapshots in `records/archive/` when restructuring.
- Update `records/run_index.md` when allocating or completing numbered task
  bundles.
