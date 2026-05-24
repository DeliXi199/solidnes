# Roadmap

## Phase 0: Project Setup

Goal: turn the research idea into a trackable project.

Deliverables:

- Project name and scope.
- Folder scaffold.
- Guidance files.
- Progress records.
- Phase 1 benchmark criteria.

Status: completed.

## Phase 0B: Implementation Strategy Selection

Goal: decide how SolidNES should enter implementation without locking into the
wrong backend too early.

Deliverables:

- Implementation strategy document.
- Reference-project notes.
- Initial backend decision record.
- Role-split Phase 1 configs.
- Backend survey.

Status: completed.

## Phase 0C: Backend Smoke-Test Preparation

Goal: prove that a selected backend route can at least build and execute a tiny
periodic solid calculation before NES-VMC work starts.

Deliverables:

- Isolated DeepSolid probe environment.
- DeepSolid config builder.
- Carbon diamond config smoke.
- Lower-cost carbon diamond runtime smoke target.
- Decision on whether DeepSolid is viable as the first backend.

Status: completed through one-iteration Adam smoke.

Exit condition:

```text
Confirm that the selected DeepSolid route can run a controlled ground-state
smoke test.
```

Completion note:

```text
DeepSolid passed CPU and CUDA GPU zero-iteration carbon diamond smoke tests on
2026-05-21, then passed a CUDA GPU one-iteration Adam smoke on job 120655.
```

## Phase 1: Carbon-Diamond Periodic NQS MVP

Goal: make the periodic ground-state NQS route reproducible and instrumented
well enough to support the first excited-state/NES-VMC extension.

Target system:

```text
C diamond primitive cell, Gamma point, paper-aligned cc-pVDZ/PBC setup.
```

Core deliverables:

- Ground-state periodic NQS run.
- Independent fixed-parameter evaluation.
- Clear backend adapter boundary.
- Pretraining diagnostics for FermiNet PBC-HF.
- First controlled two-state NES-VMC probe once the ground-state path is stable.
- Energy variance diagnostic.
- Training cost estimate.

Decision:

- Go: continue to excited-state and finite-size/twist checks.
- No-go: write a bottleneck report and revise the method.

Status: in progress. The DeepSolid and FermiNet carbon-diamond ground-state
benchmark reproduction milestone is complete. The FermiNet PBC-HF pretraining
implementation and diamond-Gamma validation milestone is also complete for the
current cc-pVDZ workflow. The active Phase 1 work is now to reproduce the
Szabo and Noe JCTC 2024 penalty-based excited-state VMC method in code and run
the first controlled NES-VMC excited-state probes.

## Phase 1B: Excited-State And Finite-Size Checks

Goal: reproduce the penalty-based excited-state VMC method in the SolidNES
code path, test whether it survives a periodic excited-state extension, and
then run basic material, finite-size, and twist checks.

Deliverables:

- Reusable penalty-based excited-state VMC objective.
- State energy, overlap, and orthogonality diagnostics.
- Carbon-diamond Gamma excited-state probe.
- Concrete material-test tasks after the controlled probe works.
- Optional twist-averaged diamond calculation.
- Finite-size and twist-sensitivity notes.

## Phase 2: First High-Value Application

Candidate A: monolayer MoS2 excitons.

Candidate B: NiO optical excitations.

Default preference after Phase 1:

```text
MoS2 before NiO, unless compute resources and magnetic-state setup are already
ready for NiO.
```

## Phase 3: Long-Term Embedding Direction

Candidate:

```text
NQS embedding solver for NV-center excited states.
```

This is a separate method-development project and should not be treated as a
core 18-month commitment.
