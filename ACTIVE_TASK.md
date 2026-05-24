# SolidNES Active Task

Last updated: 2026-05-24, Asia/Shanghai

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
Step name: Excited-state penalty-VMC method reproduction
State: configuring
Backend: FermiNet
System: carbon diamond primitive cell first, then selected material tests
Goal: reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC
      method in the SolidNES code path, then test it on concrete periodic
      materials.
```

## Current Position

```text
Benchmark reproduction is complete for both DeepSolid and FermiNet.
No Slurm jobs are currently active.
FermiNet PBC-HF pretraining is implemented, GPU-tested, and validated for the
current carbon-diamond Gamma cc-pVDZ workflow.
JAX PBC GTO target evaluation is validated for the current diamond Gamma
cc-pVDZ workflow with image cutoff 3.
Fixed-iteration 1000-step comparisons favor pretraining, but matched
short-wall-clock comparisons are mixed.
Global agent instructions live in AGENTS.md and must be read at the start of
every new answer or work session.
Numbered task bundles through 0062 are recorded in the run index. The next
available run number is 0063.
The next development target is to implement the paper-style penalty-based
excited-state VMC objective in code and run the first controlled periodic
excited-state/NES-VMC probes.
DeepQMC was cloned for source inspection under ignored `external/deepqmc/`.
The reference-source audit is recorded in
`docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
Backend-independent overlap/penalty utilities are implemented under
`src/solidnes/excited_states/`, with a no-compute synthetic validation script at
`scripts/validation/check_excited_state_penalty_objective.py`.
Future small task bundles for this phase should go under
tasks/excited_state_nesvmc/ when they produce build, smoke, experiment,
evaluation, analysis, SLURM, log, or result artifacts. Pure reference-source
audits and design notes do not consume a run number.
```

## Next Concrete Action

```text
Start the first controlled periodic excited-state/NES-VMC implementation step.

Suggested starting point:
1. Connect the backend-independent utilities to a minimal FermiNet PBC
   excited-state scaffold.
2. Define the data/interface shape for two state parameter trees and two walker
   populations.
3. Evaluate all state wavefunctions on all state samples for overlap
   diagnostics, while keeping PBC local-energy evaluation separate from
   FermiNet's molecular excited-state path.
4. Define the first minimal two-state penalty objective scaffold from
   Szabo-Noe JCTC 2024 for the FermiNet PBC backend.
5. Create the first numbered task bundle only when starting a build-only,
   smoke, training, evaluation, or analysis step that will produce project
   artifacts.
6. Keep the first probe on carbon diamond primitive Gamma, same basis/geometry.
7. Add explicit overlap/orthogonality and state-energy diagnostics before any
   production-like material test.
8. After the controlled probe works, choose concrete material tests and record
   direct-gap, indirect-gap, twist, and finite-size caveats explicitly.
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

FermiNet PBC-HF pretraining:
  records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
  records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md
  GPU pretraining target/backend probes passed in runs 0047--0050.
  Training integration and matched controls were recorded in runs 0053--0062.
```

## Completion Criteria For This Small Step

This small step is complete when all of the following are true:

```text
1. The DeepQMC/Szabo-Noe reference implementation has been inspected or
   intentionally deferred with a written reason. Done:
   `docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
2. A project-owned source-audit/design note records which parts of the method
   will be ported to SolidNES/FermiNet PBC and which parts are deferred. Done:
   `docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
3. The penalty-based excited-state VMC objective is implemented or scaffolded
   in reusable SolidNES/FermiNet code. Partial: backend-independent utilities
   exist under `src/solidnes/excited_states/`; FermiNet PBC integration is
   pending.
4. The code exposes state energies plus overlap/orthogonality diagnostics.
   Partial: overlap/orthogonality diagnostics exist; state-energy integration
   with FermiNet PBC is pending.
5. A build-only or smoke-level check proves the new code path can be imported
   and configured. Partial: no-compute synthetic utility check passed; FermiNet
   PBC build/smoke is pending.
6. A numbered task bundle is created only for the first build/smoke/run/analysis
   step that produces project artifacts.
7. The next concrete material/probe run is defined with explicit completion
   criteria.
8. The run outcome is recorded in tasks/TASK_LEDGER.md when a numbered task
   completes or materially changes.
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
