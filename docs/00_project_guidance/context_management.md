# Context Management Policy

Last updated: 2026-06-01, Asia/Shanghai

This policy keeps SolidNES project memory useful without making every Codex
session spend large token budgets on old history.

## Goals

1. Preserve historical information.
2. Keep default startup context small.
3. Make current source/config truth easy to find.
4. Avoid duplicating long narratives in many files.
5. Make archiving and slimming auditable.

## Context Layers

Hot context, read by default:

```text
AGENTS.md
CURRENT_CONTEXT.md
```

Warm context, read when directly relevant:

```text
ACTIVE_TASK.md
CURRENT_STATUS.md
PROGRESS.md
records/run_index.md
tasks/TASK_LEDGER.md
```

Cold context, search or read for history:

```text
records/progress/
records/archive/
records/decisions/
tasks/ledger/
tasks/**/README.md
tasks/**/results/
tasks/**/outputs/
tasks/**/logs/
```

## Size Targets

These are targets, not hard parser limits, but exceeding them should trigger a
split or archive:

```text
CURRENT_CONTEXT.md <= 150 lines
ACTIVE_TASK.md <= 80 lines
CURRENT_STATUS.md <= 150 lines
PROGRESS.md <= 100 lines
tasks/TASK_LEDGER.md stays a compact index
```

Large evidence, tables, plots, detailed failure analysis, and long task
timelines belong in cold context.

## What Goes Where

Use `CURRENT_CONTEXT.md` for:

- current mainline/defaults;
- source-of-truth source/config paths;
- latest milestone commit and record links;
- validation commands;
- where to look next.

Use `ACTIVE_TASK.md` for exactly one current small step:

- step name;
- state;
- current action;
- blocker;
- next action;
- completion condition.

Use `CURRENT_STATUS.md` for project-level conclusions and links to evidence.
Do not paste full task histories there.

Use `PROGRESS.md` as a short rolling update. It is not a permanent log.

Use `records/progress/YYYY-MM-DD_*.md` for permanent detailed records after
substantial work completes.

Use `tasks/TASK_LEDGER.md` as a compact index, and `tasks/ledger/` for readable
phase/task-range summaries.

Use task-local README files for detailed job IDs, configs, plots, failure modes,
and result interpretation.

## Archive Before Slimming

Before shortening or replacing a large context file, archive the original under:

```text
records/archive/YYYY-MM-DD_short_reason/original/<same-relative-path>
```

For example:

```text
records/archive/2026-06-01_context_split/original/CURRENT_STATUS.md
records/archive/2026-06-01_context_split/original/tasks/TASK_LEDGER.md
```

When the original file comes from git, verify the archive:

```text
cmp <(git show <commit>:<path>) records/archive/.../original/<path>
```

If process substitution is unavailable, use an equivalent temporary file or
checksum comparison.

## Update Pattern

For a new milestone:

1. Write the detailed record in `records/progress/`.
2. Update source/config files if behavior changes.
3. Update `CURRENT_CONTEXT.md` with only the current fact and record link.
4. Update `CURRENT_STATUS.md` only if the project-level conclusion changed.
5. Update `ACTIVE_TASK.md` only if the current small step changed.
6. Update task ledgers only when numbered task state changed.

For a completed numbered task:

1. Keep detailed artifacts in the task bundle.
2. Update the task README.
3. Update `records/run_index.md`.
4. Update compact `tasks/TASK_LEDGER.md`.
5. Update the relevant split ledger under `tasks/ledger/`.
6. Add a dated `records/progress/` note when the work is substantial.

## Staleness Checks

Before finishing a context-management change, search for stale default-read or
task-number statements:

```text
rg -n 'default.*CURRENT_STATUS|default.*PROGRESS|next available run number is 0097|Task 0096 is now active'
```

Adjust the pattern to the current migration. The goal is to catch old hot-file
rules and outdated active-task claims.

## Source Of Truth

Source code and reproducible configs define behavior. Markdown records should
summarize, cite paths, and explain decisions. They should not be the only place
where operational behavior is defined.
