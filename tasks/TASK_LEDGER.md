# SolidNES Task Ledger

Last updated: 2026-06-01, Asia/Shanghai

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
```

## Current Position

```text
Next available task number: 0106
Current source-code excited-state mainline: fused-QKV no-merge DeepQMC-aligned
Default merge_keys: []
Non-empty merge_keys: explicit comparison branches
Optional spin penalty: DeepQMC-style loss-level beta * <S^2>
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

## Maintenance Rules

- Keep this file short.
- Put detailed task history in `tasks/ledger/` and task-local README files.
- Keep exact historical snapshots in `records/archive/` when restructuring.
- Update `records/run_index.md` when allocating or completing numbered task
  bundles.
