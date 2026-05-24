# Task Bundle Numbering

Last updated: 2026-05-23

## Purpose

All run artifacts belong in numbered task bundles under `tasks/`. A task bundle
keeps result files, generated outputs, and logs together so the project does
not accumulate unrelated files in shared top-level directories.

Do not create a numbered task bundle for pure source audits, literature review,
design notes, or reference-code reading that does not produce project-owned
compute artifacts. Those notes belong in `docs/` or, for substantial completed
updates, `records/progress/`. Allocate a run number only when there is a
concrete build, smoke, experiment, evaluation, analysis, SLURM plan, backend
log, checkpoint, validation summary, or similar artifact to keep together.

Top-level `results/`, `outputs/`, and `logs/` are retired. Do not recreate them
for new work.

## Required Layout

Every task that creates compute or validation artifacts must allocate a run
number and create one bundle:

```text
tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/
  results/
  outputs/
  logs/
```

Common subfolders:

```text
results/checkpoints/
results/validation/
results/plots/
outputs/slurm_plans/
outputs/config_checks/
logs/slurm/
logs/backend/
```

Use only the subfolders needed by the task, but keep all task artifacts under
the same numbered bundle.

## Numbering Rule

Run numbers are global across all task categories. They are not restarted per
folder. Use four digits: `0001`, `0002`, and so on.

Before creating a task:

```text
1. Read records/run_index.md.
2. Allocate the next unused NNNN number.
3. Create the task bundle.
4. Point experiment YAML output paths into that bundle.
5. Add or update records/run_index.md.
6. Update tasks/TASK_LEDGER.md when the task completes or materially changes.
```

Run numbers are never reused, even if a job fails or a retired task is deleted.

## Config Rule

Experiment YAML output paths should point inside the task bundle:

```yaml
output:
  log_dir: tasks/phase1_diamond_c/pbc_gamma/training/NNNN_short_slug/logs/backend
  checkpoint_dir: tasks/phase1_diamond_c/pbc_gamma/training/NNNN_short_slug/results/checkpoints
  validation_dir: tasks/phase1_diamond_c/pbc_gamma/training/NNNN_short_slug/results/validation
```

Restore paths should also prefer existing task-bundle checkpoint directories.
Only use historical paths when the target has not yet been migrated.

## SLURM Outputs

Generated SLURM plans:

```text
tasks/.../NNNN_short_slug/outputs/slurm_plans/plan.json
```

SLURM logs:

```text
tasks/.../NNNN_short_slug/logs/slurm/NNNN_short_slug_<jobid>.log
tasks/.../NNNN_short_slug/logs/slurm/NNNN_short_slug_<jobid>.err
```

For the existing SLURM planner, set:

```bash
SOLIDNES_JOB_NAME=NNNN_short_slug
SOLIDNES_PLAN_JSON=tasks/.../NNNN_short_slug/outputs/slurm_plans/plan.json
SOLIDNES_SLURM_LOG_DIR=tasks/.../NNNN_short_slug/logs/slurm
```

The planner writes logs using `%x_%j`, so the job name must include the run
number.

## Ledgers

- `records/run_index.md`: canonical run-number ledger and next available number.
- `tasks/TASK_LEDGER.md`: readable task purpose and result ledger.
- `tasks/MIGRATION_MANIFEST.tsv`: machine-readable legacy migration map.

## Completion Rule

A task is not fully recorded until all of the following are true:

```text
1. records/run_index.md has the run number and task root.
2. Results, generated outputs, and logs are under the same task bundle.
3. tasks/TASK_LEDGER.md records what was done and the key result.
4. A dated progress record links to the task bundle when the work is substantial.
```
