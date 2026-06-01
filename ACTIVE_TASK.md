# SolidNES Active Task

Last updated: 2026-06-01, Asia/Shanghai

This file records only the current small step. Historical task state belongs in
`records/progress/`, `tasks/TASK_LEDGER.md`, and task-local README files.

## Current Small Step

```text
Step name: 0105 DeepQMC state-specific spin GPU smoke
State: complete
Current action: SLURM job `135764` completed successfully via
  `scripts/slurm/submit_ferminet_gpu_smoke.sh` on `amdgpu40g/gpu004`.
Current blocker: none.
Next action: final diff review, update FermiNet patch/docs, then commit/push.
Completion condition: satisfied; train_stats and energy/overlap/S2 diagnostic
  arrays are finite and records document state-specific spin alignment.
```

## Current Project Position

```text
Mainline excited-state method:
  DeepQMC-aligned no-merge PsiFormer/FermiNet native vmc_overlap.

Default:
  merge_keys: []

Non-default merge:
  supported as explicit merge_key_variant comparison branches.

Latest source milestone:
  34d6574 Set no-merge excited-state mainline
  916bcc4 Record no-merge excited-state milestone

Latest spin-penalty update:
  records/progress/2026-06-01_spin_penalty_deepqmc_alignment.md

Current hot context:
  CURRENT_CONTEXT.md

Next available task number:
  0106
```
