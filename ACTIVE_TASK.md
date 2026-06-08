# SolidNES Active Task

Last updated: 2026-06-08, Asia/Shanghai

This file records only the current small step. Historical task state belongs in
`records/progress/`, `tasks/TASK_LEDGER.md`, and task-local README files.

## Current Small Step

```text
Step name: 2026-06-08 spin-penalty/default-lr milestone save
State: completed
Current action: milestone record and default-parameter data snapshot prepared
  for Git commit and GitHub push.
Current blocker: none.
Next action: use `eta0=0.02`, `tau=10000`, `learning_rate_decay=1.0`, and
  `kfac.damping=0.001` as the default optimizer schedule for future
  excited-state calculations unless a new task explicitly overrides it.
Completion condition: milestone record and data snapshot are committed and
  pushed to GitHub.
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
  spin penalty implementation complete
  default excited-state optimizer schedule selected from 0113/0114

Default excited-state optimizer schedule:
  eta0=0.02, tau=10000, learning_rate_decay=1.0
  kfac.damping=0.001

Latest spin-penalty update:
  records/progress/2026-06-08_spin_penalty_default_lr_milestone.md
  records/progress/2026-06-01_spin_penalty_deepqmc_alignment.md

Current hot context:
  CURRENT_CONTEXT.md

Next available task number:
  0115
```
