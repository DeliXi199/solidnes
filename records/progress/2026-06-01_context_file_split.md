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
```

Archived copies include:

```text
AGENTS.md
CURRENT_STATUS.md
ACTIVE_TASK.md
PROGRESS.md
TASK_LEDGER.md
run_index.md
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
