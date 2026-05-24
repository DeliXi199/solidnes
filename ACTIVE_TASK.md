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
Numbered task bundles through 0063 are recorded in the run index. The next
available run number is 0064.
The next development target is to implement the paper-style penalty-based
excited-state VMC objective in code and run the first controlled periodic
excited-state/NES-VMC probes.
DeepQMC was cloned for source inspection under ignored `external/deepqmc/`.
The reference-source audit is recorded in
`docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
Backend-independent overlap/penalty utilities are implemented under
`src/solidnes/excited_states/`, with a no-compute synthetic validation script at
`scripts/validation/check_excited_state_penalty_objective.py`.
A minimal FermiNet PBC excited-state scaffold is implemented at
`src/solidnes/excited_states/ferminet_pbc_scaffold.py`, with a no-compute
synthetic validation script at
`scripts/validation/check_ferminet_pbc_excited_scaffold.py`.
The first real FermiNet/JAX build-only adapter check is implemented at
`scripts/validation/check_ferminet_pbc_excited_adapter_build.py` and passed in
the `solidnes-ferminet-jax0101-cuda12` environment. It keeps
`cfg.system.states == 0`, initializes two external state parameter trees, wraps
`network.apply` into the scaffold wavefunction-matrix interface, and constructs
the PBC local-energy wrapper without evaluating the expensive Laplacian path by
default.
The FermiNet/JAX wrapper pattern has been promoted from the validation script
into reusable source at
`src/solidnes/excited_states/ferminet_pbc_adapter.py`. The validation script now
calls this module instead of carrying a duplicate implementation.
The real FermiNet adapter is now connected to the penalty objective through
`evaluate_ferminet_pbc_penalty_terms(...)`, which returns state energies,
wavefunction ratios, overlap diagnostics, and penalty terms. The build-only
validation script
`scripts/validation/check_ferminet_pbc_penalty_terms.py` passed with a cheap
local-energy stand-in.
The first differentiable optimization-step scaffold is implemented through
`value_and_grad_ferminet_pbc_penalty_objective(...)` and
`apply_external_state_sgd_step(...)`. The build-only validation script
`scripts/validation/check_ferminet_pbc_penalty_grad_step.py` passed with a
cheap local-energy stand-in and confirmed finite nonzero gradients plus a real
parameter update.
The first build-only multi-step optimization smoke is implemented at
`scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py`. It uses the real
FermiNet PBC external-state adapter with a cheap local-energy stand-in and
confirmed three consecutive finite SGD updates with decreasing penalty
objective, without creating a numbered task bundle.
Future small task bundles for this phase should go under
tasks/excited_state_nesvmc/ when they produce build, smoke, experiment,
evaluation, analysis, SLURM, log, or result artifacts. Pure reference-source
audits and design notes do not consume a run number.
Run 0063 is allocated for the first scheduled real PBC local-energy/Laplacian
smoke:
`tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/`.
The dry-run plan passed and selected one A100 80GB GPU on `intelgpu80g`,
node `gpu001`, with 8 CPU cores and a 20-minute walltime.
Run 0063 was first submitted as Slurm job `128435` with 1 GPU and 8 CPU
cores; it completed successfully. Per user request, the task was resubmitted
as full-node Slurm job `128439` on `intelgpu80g/gpu001`, using both A100 80GB
GPUs, 96 CPU cores, and an exclusive allocation.
The full-node real PBC local-energy/Laplacian smoke passed. Job `128439`
completed in 00:02:05 with exit code `0:0`; JAX reported `cuda:0` and
`cuda:1`; the validation summary recorded finite `[2, 1]` local energies,
finite `[2]` state energies, a finite `2x2` overlap matrix, and a finite
scalar penalty objective.
The first reusable fixed-sample training-loop helper is implemented at
`src/solidnes/excited_states/ferminet_pbc_training.py`. It owns external
state-parameter SGD updates, returns per-step penalty/energy/overlap/collapse
diagnostics, and is verified through
`scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py` with a cheap
local-energy stand-in.
```

## Next Concrete Action

```text
Decide the first real-local-energy multi-step training-loop smoke.

Suggested starting point:
1. Keep `cfg.system.states == 0` and continue managing state trees outside
   upstream FermiNet PBC.
2. Reuse the real PBC local-energy entry point validated by run 0063.
3. Use the new fixed-sample training-loop helper for a tiny real-local-energy
   multi-step smoke.
4. Allocate run 0064 only if that smoke is scheduled or produces durable
   Slurm/log/result artifacts.
5. If it passes, start adding sampler/checkpoint/output integration around the
   training-loop helper.
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
   and a minimal FermiNet PBC scaffold exist under
   `src/solidnes/excited_states/`; reusable FermiNet/JAX PBC adapter wrappers
   and a fixed-sample external-state training-loop helper exist, but sampler,
   checkpoint, and production-driver integration are pending.
4. The code exposes state energies plus overlap/orthogonality diagnostics.
   Partial: scaffold-level and adapter-level state-energy/overlap/penalty
   diagnostics exist; real local-energy evaluation passed in run 0063, but
   reusable training integration is pending.
5. A build-only or smoke-level check proves the new code path can be imported
   and configured. Done for build-only: synthetic utility/scaffold checks,
   the FermiNet/JAX adapter build check, and the cheap-local-energy
   FermiNet/JAX penalty-term, gradient-step, and multi-step optimization
   checks passed. Done for scheduled GPU smoke: run 0063 passed the real PBC
   local-energy/Laplacian check with two external state parameter trees.
   Done for source-level training integration: the cheap-local-energy
   multi-step smoke now exercises the reusable training-loop helper.
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
