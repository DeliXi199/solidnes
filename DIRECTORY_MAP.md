# Directory Map

This file defines what belongs in each top-level folder.

## Top-Level Files

- `README.md`: short project entry point.
- `AGENTS.md`: global operating instructions for any assistant or coding
  agent; read first at the start of each new answer or work session.
- `PROJECT_GUIDE.md`: project identity, scope, principles, and operating rules.
- `CURRENT_STATUS.md`: short current-state snapshot, milestone summary,
  evidence jobs, caveats, and next phase.
- `ACTIVE_TASK.md`: exact current small step, next action, active job state,
  expected output, and completion criteria.
- `PROGRESS.md`: short rolling current-state snapshot and immediate next
  actions; detailed history belongs in `records/progress/`.
- `ROADMAP.md`: phase-level plan.
- `DIRECTORY_MAP.md`: repository structure and file placement rules.

## `docs/`

Long-lived human-facing project documents.

Use for:

- Guidance notes.
- Literature notes.
- Theory notes.
- Meeting notes.
- Reports and draft outlines.
- Reference-project architecture notes.

Do not use for:

- Raw training logs.
- Large generated results.
- Temporary scratch outputs.

## `records/`

Chronological project memory.

Use for:

- Dated progress notes.
- Decision records.
- Human-readable experiment logs.
- `run_index.md`, the canonical ledger for numbered task bundles.

This folder answers:

```text
What happened, when, and why?
```

## `configs/`

Reproducible experiment settings.

Each serious experiment should have a config file before it has a run script.

Configs are split by role:

```text
experiment/
system/
model/
sampler/
train/
```

## `experiments/`

Experiment-specific run and analysis folders.

Each experiment folder should include a README describing:

- Scientific purpose.
- Target system.
- Configs used.
- Expected outputs.
- Current status.

## `src/solidnes/`

Python source package.

Use for:

- Backend adapter modules.
- Smoke-only compatibility shims.
- Future reusable NES-VMC implementation code.

Keep command-line glue in `scripts/`; keep reusable logic here.

## `tasks/`

Numbered task bundles for runs, analyses, probes, and evaluations.

Every new run/task should create one bundle:

```text
tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/
```

Common task-type folders include `smoke/`, `training/`, `continuation/`,
`evaluation/`, `ablation/`, `references/`, and `analysis/`. Allocate `NNNN`
in `records/run_index.md` before creating a new task bundle.

FermiNet PBC-HF pretraining tasks are grouped separately under:

```text
tasks/ferminet_pretraining/NNNN_short_slug/
```

Excited-state/NES-VMC implementation and material-test tasks are grouped under:

```text
tasks/excited_state_nesvmc/NNNN_short_slug/
```

Use this area for reproducing the Szabo and Noe JCTC 2024 penalty-based
excited-state VMC method in code, then testing the resulting workflow on
specific periodic materials.

Inside each task bundle, keep generated files together:

```text
tasks/.../NNNN_short_slug/results/
tasks/.../NNNN_short_slug/outputs/
tasks/.../NNNN_short_slug/logs/
```

Use `tasks/TASK_LEDGER.md` as the readable task ledger. Use
`tasks/MIGRATION_MANIFEST.tsv` as the machine-readable migration map.

Top-level `results/`, `outputs/`, and `logs/` are retired. Do not recreate
them for new work.

## `references/`

Local literature notes, citation metadata, and curated reference material.

## `external/`

Local checkouts or copies of upstream backend projects used for inspection,
compatibility probes, and controlled patch development.

Do not treat this as SolidNES-owned source. Keep SolidNES reusable code under
`src/solidnes/`; keep durable external-backend changes as patch files under
`patches/`.

## `patches/`

Patch records for external backend changes.

Use for:

- Diffs against upstream backend files.
- Notes needed to reapply a local compatibility fix.
- Small backend-specific patch documentation.

Do not use for generated results, logs, or experiment outputs.

## `scripts/`

Small utility scripts and cluster launchers.

Scripts should call configs rather than hard-code experiment details.

Use `scripts/slurm/` for SLURM planners, submit wrappers, and job templates.

New generated SLURM plans belong under a task bundle's
`outputs/slurm_plans/`; new generated SLURM logs belong under the same task
bundle's `logs/slurm/`.

Use `scripts/maintenance/` for repository and task-layout checks.

## `.venv/`

Local disposable Python virtual environment. It is ignored by git and can be
rebuilt from environment notes when needed.
