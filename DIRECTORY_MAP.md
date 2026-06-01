# Directory Map

This file defines what belongs in each top-level folder.

## Top-Level Files

- `README.md`: short project entry point.
- `AGENTS.md`: global operating instructions for any assistant or coding
  agent; read first at the start of each new answer or work session.
- `PROJECT_GUIDE.md`: project identity, scope, principles, and operating rules.
- `CURRENT_CONTEXT.md`: short hot startup context with current defaults,
  source-of-truth paths, latest milestone, and where to look next.
- `CURRENT_STATUS.md`: short project-level state snapshot, milestone summary,
  evidence jobs, caveats, and next phase.
- `ACTIVE_TASK.md`: exact current small step, next action, active job state,
  expected output, and completion criteria only.
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
- Reusable NES-VMC implementation code such as
  `src/solidnes/excited_states/`.

Keep command-line glue in `scripts/`; keep reusable logic here.

## `tasks/`

Numbered task bundles for runs, analyses, probes, and evaluations that produce
project artifacts.

Every new run/task that creates build, smoke, experiment, evaluation, analysis,
SLURM, log, result, or validation artifacts should create one bundle:

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

PsiFormer/self-attention implementation and benchmark tasks are grouped under:

```text
tasks/psiformer/NNNN_short_slug/
```

Use this area for self-attention model build checks, speed probes, and
PsiFormer-vs-FermiNet comparison artifacts.

Do not create a numbered task bundle for pure source audits, literature review,
external-code inspection, or design notes. Keep those under `docs/` or
`records/progress/`.

Inside each task bundle, keep generated files together:

```text
tasks/.../NNNN_short_slug/results/
tasks/.../NNNN_short_slug/outputs/
tasks/.../NNNN_short_slug/logs/
```

For comparative sweeps, ablations, or matched controls, use one task bundle for
the whole comparison and place variants underneath it:

```text
tasks/.../NNNN_short_slug/runs/<variant>/results/
tasks/.../NNNN_short_slug/runs/<variant>/outputs/
tasks/.../NNNN_short_slug/runs/<variant>/logs/
tasks/.../NNNN_short_slug/results/validation/
```

The parent `results/validation/` directory holds cross-variant tables and
plots. Individual variant summaries stay inside their `runs/<variant>/`
directories.

Use `tasks/TASK_LEDGER.md` as the compact task-ledger index and
`tasks/ledger/` for split readable ledgers. Use `tasks/MIGRATION_MANIFEST.tsv`
as the machine-readable migration map.

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

Use `scripts/validation/` for lightweight validation and summary scripts,
including no-compute synthetic checks for reusable source utilities.

## `.venv/`

Local disposable Python virtual environment. It is ignored by git and can be
rebuilt from environment notes when needed.
