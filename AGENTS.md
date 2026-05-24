# SolidNES Agent Instructions

This file is the global operating instruction for any assistant or coding agent
working in this repository.

Read this file at the start of every new answer or work session before taking
actions in the project.

## Mandatory Startup Read

At the beginning of every new answer or resumed work session, read these files
in this order:

```text
1. AGENTS.md
2. CURRENT_STATUS.md
3. ACTIVE_TASK.md
4. PROGRESS.md, only the relevant current/recent section unless deeper history
   is needed
5. DIRECTORY_MAP.md, when creating or moving files
```

If a task involves SLURM, CPU jobs, GPU jobs, queue status, or submitted
experiments, also read:

```text
docs/00_project_guidance/slurm_task_management.md
```

If a task creates compute-result output files or new result directories, also
read:

```text
docs/00_project_guidance/result_output_numbering.md
records/run_index.md
tasks/TASK_LEDGER.md
```

If the user asks what the project has achieved, answer from
`CURRENT_STATUS.md`. If the user asks what exact small step is active, answer
from `ACTIVE_TASK.md`.

## Current Project Truth

The current high-level truth is maintained in:

```text
CURRENT_STATUS.md
```

The current small-step truth is maintained in:

```text
ACTIVE_TASK.md
```

Do not rely only on chat memory for project state. The files above are the
handoff source of truth.

## State Recording Rules

Use the files as follows:

```text
CURRENT_STATUS.md
  Project-level snapshot: current conclusion, major milestones, evidence jobs,
  caveats, and next phase.

ACTIVE_TASK.md
  Exact small step: state, current action, next action, active job id, expected
  output, and completion criteria.

PROGRESS.md
  Short rolling current progress plus important recent updates. Detailed
  history belongs in records/progress/.

records/progress/YYYY-MM-DD_*.md
  Permanent dated record after a task, run, analysis, or decision completes.

records/decisions/*.md
  Strategic decisions and rationale.

records/run_index.md
  Numbered task-bundle ledger. Allocate every new run/task bundle from this
  file.

tasks/TASK_LEDGER.md
  Human-readable ledger of numbered task purposes, locations, and key
  outcomes. Update it when a task completes or materially changes.
```

Every time a job is submitted, starts running, completes, fails, or its results
are analyzed, update `ACTIVE_TASK.md`. When the step is complete, write or
update a dated record under `records/progress/` and update
`tasks/TASK_LEDGER.md`.

## Current Milestone

Two Phase 1 enabling milestones are complete at the project-validation level:

```text
1. Carbon-diamond paper benchmark reproduction through both DeepSolid and
   FermiNet.
2. FermiNet PBC-HF pretraining implementation and diamond-Gamma validation for
   the current cc-pVDZ workflow.
```

Evidence:

```text
DeepSolid direct route:
  Job 127816
  Tail-2000 mean: -75.4161279970 Ha
  DeepSolid supplementary diamond reference: -75.4009 Ha

FermiNet route:
  Training job 127898
  Fixed-parameter evaluation job 127992
  Evaluation mean: -75.4125655570 Ha
  DeepSolid supplementary diamond reference: -75.4009 Ha

FermiNet PBC-HF pretraining:
  Runs 0047--0050
  JAX PBC GTO cc-pVDZ AO max abs error: 1.12e-9
  JAX PBC GTO cc-pVDZ occupied-MO max abs error: 8.51e-10
  Runs 0053--0062
  Fixed-iteration early training favored pretraining, while short wall-clock
  timeboxed comparisons were mixed.
```

Current next phase:

```text
Reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC method
in code, then test it on concrete periodic materials.
```

## SLURM And Scheduler Rules

Never submit jobs casually.

Before submitting any CPU or GPU job:

```text
1. Read docs/00_project_guidance/slurm_task_management.md.
2. Build or inspect the experiment config.
3. Run build-only or no-compute verification when available.
4. Run the approved submitter in dry-run mode.
5. Inspect the generated plan in the task bundle under `outputs/slurm_plans/`.
6. Confirm partition, node, GPU count, CPU count, exclusivity, walltime, and
   blocked partitions are appropriate.
7. Update ACTIVE_TASK.md before submission.
```

Approved submitters:

```text
FermiNet GPU:
  scripts/slurm/submit_ferminet_gpu_smoke.sh

FermiNet CPU config/build check:
  scripts/slurm/submit_ferminet_cpu_config_check.sh

DeepSolid GPU:
  scripts/slurm/submit_deepsolid_gpu_smoke.sh

DeepSolid CPU:
  scripts/slurm/submit_deepsolid_cpu_smoke.sh
```

Do not use one-off `sbatch` commands unless the user explicitly requests it and
the scheduler policy cannot support the job shape.

Project submitters must target a task bundle. Set `SOLIDNES_TASK_ROOT` unless
the submitter call intentionally provides both an explicit plan path and an
explicit SLURM log directory.

GPU rules:

```text
- GPU compute must not go to the test partition by default.
- The test partition is allowed only for tiny feasibility checks when explicitly
  requested and configured.
- Always dry-run first.
- Prefer the project submitters so plans and logs are generated.
- Keep generated plans inside the active task bundle under
  `tasks/.../outputs/slurm_plans/`.
- Keep SLURM logs inside the active task bundle under `tasks/.../logs/slurm/`.
- For current FermiNet production-like runs, follow the established GPU
  scheduling path and resource rules instead of manually choosing resources.
```

CPU rules:

```text
- CPU-only config/build checks and documentation-producing analysis can use CPU
  resources.
- Short CPU jobs may use test only according to the scheduler policy.
- Any JAX model execution, one-iteration training, realistic VMC, or NES-VMC
  optimization is a GPU job unless explicitly designed as a no-compute check.
```

## Experiment Rules

Before running a serious experiment:

```text
1. Make sure the experiment has a YAML config.
2. Read docs/00_project_guidance/result_output_numbering.md.
3. Allocate or confirm a run number in records/run_index.md.
4. Create or target one task bundle:
   tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/.
5. Put new result files inside `tasks/.../NNNN_short_slug/results/`.
6. Put generated planner/output files inside `tasks/.../NNNN_short_slug/outputs/`.
7. Put SLURM and backend logs inside `tasks/.../NNNN_short_slug/logs/`.
8. Make sure the experiment output paths all point inside the same task bundle.
9. Make sure the purpose and completion criteria are clear in ACTIVE_TASK.md.
10. Update tasks/TASK_LEDGER.md when the task completes or materially changes.
```

Do not create new mixed unnumbered output folders. Preserve task-type grouping
such as `training/`, `continuation/`, `evaluation/`, `ablation/`, `smoke/`,
`references/`, and `analysis/`; the final task bundle under that task-type
directory must start with the zero-padded run number. Keep `results/`,
`outputs/`, and `logs/` inside that bundle so files do not pile up in shared
top-level folders.

After a run finishes:

```text
1. Check sacct/squeue state.
2. Check stdout and stderr logs.
3. Check train_stats.csv or the relevant result file.
4. Generate or update validation summaries.
5. Generate plots when the user asks or when they clarify interpretation.
6. Update ACTIVE_TASK.md.
7. Update tasks/TASK_LEDGER.md.
8. Write or update a dated record under records/progress/.
9. Update CURRENT_STATUS.md only if the run changes a milestone or project
   conclusion.
```

## Interpretation Rules

Be precise about what a result proves.

Use these labels:

```text
training-chain statistic
fixed-parameter evaluation
smoke/runtime check
build-only check
paper-like benchmark
production benchmark
```

Do not call a training-chain statistic a final fixed-wavefunction estimate.
Do not call a smoke test a physics result.

Current accepted statement:

```text
DeepSolid direct calculation and FermiNet calculation both reached or exceeded
the DeepSolid supplementary diamond reference energy for the carbon-diamond
benchmark. FermiNet has an independent fixed-parameter evaluation; DeepSolid
job 127816 is currently training-chain evidence and can be made cleaner with a
fixed-checkpoint evaluation if needed.
```

## Coding Rules

Follow the repository style:

```text
- Prefer existing config/script/adapter patterns.
- Keep experiment settings in YAML, not hard-coded in scripts.
- Use apply_patch for manual edits.
- Do not rewrite unrelated files.
- Do not remove user or previous-run outputs.
- Keep generated files for new tasks under the appropriate `tasks/.../NNNN.../`
  bundle.
```

When adding files, check `DIRECTORY_MAP.md` first.

## Next-Step Default

Unless the user gives a different instruction, the next technical step is:

```text
Prepare a short FermiNet GPU PBC-HF pretraining pilot, verify it by dry-run
under the scheduler policy, submit it only through the approved FermiNet
submitter, and compare its early VMC behavior against the no-pretraining x64
baseline.
```
