# 2026-06-01 Context File Split

The project context files were reorganized into a hot/cold structure so Codex
does not need to read large historical records at every startup.

## Hot Context

Default startup is now:

```text
AGENTS.md
CURRENT_CONTEXT.md
```

`CURRENT_CONTEXT.md` carries the current source defaults, source-of-truth paths,
latest milestone, validation commands, and where to look next.

The ongoing policy is recorded in:

```text
docs/00_project_guidance/context_management.md
```

## Cold Context

Historical detail remains available through:

```text
CURRENT_STATUS.md
ACTIVE_TASK.md
PROGRESS.md
records/progress/
records/archive/
records/run_index.md
tasks/TASK_LEDGER.md
tasks/ledger/
tasks/**/README.md
```

The exact pre-split files were archived under:

```text
records/archive/2026-06-01_context_split/
records/archive/2026-06-01_context_split/original/
```

Verified original copies exported from commit `916bcc4` include:

```text
original/AGENTS.md
original/CURRENT_STATUS.md
original/ACTIVE_TASK.md
original/PROGRESS.md
original/README.md
original/PROJECT_GUIDE.md
original/DIRECTORY_MAP.md
original/docs/00_project_guidance/research_workflow.md
original/docs/00_project_guidance/result_output_numbering.md
original/records/run_index.md
original/tasks/TASK_LEDGER.md
```

## Split Ledgers

`tasks/TASK_LEDGER.md` is now a compact index. Readable slices live under:

```text
tasks/ledger/0001_0062_ground_pretrain.md
tasks/ledger/0063_0093_ferminet_excited.md
tasks/ledger/0094_0103_psiformer_deepqmc.md
```

## Non-Destructive Guarantee

The migration was intentionally non-destructive:

- old large files were copied to `records/archive/2026-06-01_context_split/`
  before live files were shortened;
- task-local README files and dated progress records remain in place;
- `records/run_index.md` still lists numbered task roots;
- git history preserves the prior versions.
