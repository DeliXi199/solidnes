# Progress

Last updated: 2026-06-01

This is a short rolling summary. Detailed historical progress is stored under
`records/progress/` and task-local README files.

## Current State

The DeepQMC-aligned excited-state method is now the source-code mainline:

```text
PsiFormer/FermiNet native vmc_overlap
fused_qkv attention
independent per-state parameter trees
merge_keys: []
diagonal independent-state fast paths enabled
```

Non-empty `merge_keys` remain implemented for explicit comparison branches.

## Recent Changes

- `34d6574`: set the no-merge excited-state mainline in source/config defaults.
- `916bcc4`: recorded the no-merge excited-state milestone in project files.
- `0103`: completed the attention x merge-key 10000-step comparison bundle and
  generated energy/gap plots.
- `0102`: validated diagonal fast paths at 4GPU batch4096 scale; stable step
  time was 1.2606 s with diagonal paths versus 2.1733 s without.
- Project policy remains: iterative training/evaluation tasks with
  `iterations >= 1000` must save the final-step checkpoint.

## Active Step

See `ACTIVE_TASK.md`.

Current short version:

```text
Context-file split is in progress.
Goal: make AGENTS.md + CURRENT_CONTEXT.md enough for default Codex startup.
Historical context remains archived and searchable.
```

## History Links

```text
CURRENT_CONTEXT.md
CURRENT_STATUS.md
records/progress/
records/archive/2026-06-01_context_split/
tasks/TASK_LEDGER.md
tasks/ledger/
tasks/**/README.md
```
