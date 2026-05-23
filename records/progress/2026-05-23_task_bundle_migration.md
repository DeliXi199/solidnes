# Task Bundle Migration

Date: 2026-05-23, Asia/Shanghai

## Purpose

Generated artifacts were migrated from separate top-level generated folders into
numbered task bundles under `tasks/`.

Required layout:

```text
tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/
  results/
  outputs/
  logs/
```

## Completed Migration

The global migration created numbered task bundles for migrated artifacts and
assigned run numbers by first artifact timestamp.

Current migration records:

- `tasks/MIGRATION_MANIFEST.tsv`: machine-readable migration map.
- `records/run_index.md`: run-number ledger and next available number.
- `tasks/TASK_LEDGER.md`: human-readable task purpose and result ledger.

## Follow-Up Cleanup

The previous partial task-tree backup was verified and then removed. The
retired target-specific scaffold and its task bundle were also removed after
the project focus was confirmed as carbon-diamond-centered.

## Current Rule

Do not write new artifacts to top-level `results/`, `outputs/`, or `logs/`.
Every new task must allocate the next global run number and write all artifacts
inside its numbered bundle.
