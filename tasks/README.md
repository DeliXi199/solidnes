# Task Bundles

This directory contains globally numbered task bundles migrated from legacy `results/`, `outputs/`, and `logs/`.

Use `TASK_LEDGER.md` as the human-facing task ledger. Every completed or materially updated numbered task should update that file with what was done, what result was obtained, and the key artifact links.

Each task bundle uses this layout:

```text
tasks/<classification>/<task_type>/NNNN_task_name/
  logs/
  outputs/
  results/
```

Numbers are assigned by the first available artifact timestamp across all migrated tasks, not per classification folder.

Current high-value classifications:

```text
phase1_diamond_c/sto3g/
phase1_diamond_c/ccpvdz/
phase1_diamond_c/pbc_gamma/
```

Keep generated files inside the numbered task bundle:

- `results/`: checkpoints, validation summaries, plots, analysis artifacts.
- `outputs/`: scheduler plans, build/config checks, deterministic generated files.
- `logs/`: SLURM stdout/stderr and backend runtime logs.

Do not recreate top-level `results/`, `outputs/`, or `logs/` for new work.
