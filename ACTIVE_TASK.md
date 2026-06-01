# SolidNES Active Task

Last updated: 2026-06-01, Asia/Shanghai

This file records only the current small step. Historical task state belongs in
`records/progress/`, `tasks/TASK_LEDGER.md`, and task-local README files.

## Current Small Step

```text
Step name: Context-file split and hot-start cleanup
State: complete
Current action: no compute job is running. Large default-read context files have
  been replaced with the short CURRENT_CONTEXT.md entry point; old full files
  are archived under records/archive/2026-06-01_context_split/.
Current blocker: none.
Next action: use AGENTS.md + CURRENT_CONTEXT.md as the default startup context;
  read deeper status/history files only when needed.
Completion condition: AGENTS.md defaults to AGENTS.md + CURRENT_CONTEXT.md;
  active/status/progress files are short; full old context files are archived;
  git diff passes whitespace checks; commit is pushed to origin/main.
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

Current hot context:
  CURRENT_CONTEXT.md

Next available task number:
  0104
```
