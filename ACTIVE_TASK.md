# SolidNES Active Task

Last updated: 2026-05-23, Asia/Shanghai

## Purpose

This file records the exact small step currently in progress.

Use it as the handoff point when asking:

```text
Now what is the task doing?
What is the next concrete action?
Is a Slurm job running?
What result are we waiting for?
What condition marks this step complete?
```

Long-term milestones belong in `CURRENT_STATUS.md`. Chronological completed
work belongs in `records/progress/`.

## Current Small Step

```text
Step name: FermiNet PBC-HF pretraining validation
State: ready_to_start
Backend: FermiNet
System: carbon diamond primitive cell
Goal: verify that PBC-HF pretraining works on GPU and improves the early VMC
      trajectory against the no-pretraining x64 baseline.
```

## Current Position

```text
Benchmark reproduction is complete for both DeepSolid and FermiNet.
No Slurm jobs are currently active.
The next development target is FermiNet PBC-HF pretraining.
PBC-HF pretraining code exists and has passed build/local CPU probes.
Global agent instructions now live in AGENTS.md and must be read at the start
of every new answer or work session.
New run artifacts must use one task bundle per task. The bundle is grouped by
task type and contains its own `results/`, `outputs/`, and `logs/` folders.
Numbered task bundles through 0046 are migrated where retained. The retired
target-specific scaffold was removed from the tree, and the next available run
number is 0047.
Run number 0045 is the current FermiNet PBC-HF pretraining GPU pilot bundle.
The existing PBC-HF pretraining experiment YAML now writes to the typed
numbered task bundle for run 0045, and a FermiNet config build check passed
after the bundle-layout change.
The global migration moved legacy artifacts into numbered task bundles and
retired top-level generated artifact directories.
The next small step is not a production training run yet; it is a controlled
GPU pretraining pilot plus diagnostics check.
```

## Next Concrete Action

```text
Prepare and submit a short GPU PBC-HF pretraining pilot through:
scripts/slurm/submit_ferminet_gpu_smoke.sh

Dry-run command:
SOLIDNES_TASK_ROOT=tasks/ferminet_pretraining/0045_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml \
SOLIDNES_JOB_NAME=0045_ferminet_pbc_hf_pretrain_pilot \
SOLIDNES_DRY_RUN=1 \
bash scripts/slurm/submit_ferminet_gpu_smoke.sh

Before real submission:
1. Inspect the generated plan under the task bundle's outputs/slurm_plans/.
2. Verify the selected GPU partition is not test.
3. Confirm requested GPU/CPU counts obey the project scheduling rule.
4. Update ACTIVE_TASK.md with the dry-run result.
```

## Active Or Pending Jobs

```text
None.
```

## Evidence Already Available

```text
DeepSolid reproduction:
  Job 127816
  Tail-2000 mean: -75.4161279970 Ha
  Paper reference: -75.4009 Ha

FermiNet reproduction:
  Training job 127898
  Evaluation job 127992
  Fixed-parameter evaluation mean: -75.4125655570 Ha
  Paper reference: -75.4009 Ha

FermiNet PBC-HF pretraining implementation:
  records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
  Build-only checks passed.
  Local CPU one-step PBC pretraining probe passed with a sto-3g target.
```

## Completion Criteria For This Small Step

This small step is complete when all of the following are true:

```text
1. A GPU PBC-HF pretraining pilot has completed without traceback.
2. Logs show PBC-HF pretraining actually ran, not silently skipped.
3. The run records pretraining diagnostics clearly enough to judge behavior.
4. The resulting early VMC trajectory can be compared with the no-pretraining
   x64 baseline.
5. Results, outputs, and logs are all under the run 0045 task bundle.
6. The task bundle has enough local metadata to identify what was run.
7. A dated record is written under records/progress/.
8. The run outcome is recorded in tasks/TASK_LEDGER.md.
```

## What To Record On Every Update

For each change of state, update only this file and then write permanent details
to a dated record when the step completes.

Use these fields:

```text
State:
  ready_to_start | configuring | dry_run_done | submitted | running |
  completed | failed | blocked

Current action:
  One sentence describing what is happening now.

Next action:
  The next concrete command or decision.

Job ID:
  Slurm job id, or None.

Run ID:
  Number from records/run_index.md, or None.

Task ledger:
  Whether tasks/TASK_LEDGER.md needs to be updated.

Expected output:
  The file or metric we are waiting for.

Completion condition:
  The exact condition that lets us move to the next step.
```

## State Transition Rule

```text
Before submitting a job:
  State -> configuring or dry_run_done

After sbatch returns a job id:
  State -> submitted

After the job starts:
  State -> running

After analysis and plots are written:
  State -> completed

If a traceback, bad scheduler request, or missing result appears:
  State -> failed or blocked, with the reason written here.
```
