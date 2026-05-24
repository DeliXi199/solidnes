# Project Guide

## Project Identity

Project name: SolidNES

Full name: Neural Quantum States for Excited States of Periodic Solids

Core question:

Can neural quantum states compute excited states of periodic solids from first
principles without excited-state training labels?

## Scope

The project starts as a method-development and benchmark project, not as a broad
materials application project.

The first scientific target is:

```text
Periodic-solid NQS method development on carbon diamond.
```

The goal is not to claim a new materials result immediately. The goal is to
make the periodic NQS workflow stable, interpretable, reproducible, and ready
for controlled excited-state extensions.

## Phase 1 MVP

The MVP is successful only if the project can produce:

- A reproducible periodic NQS ground-state workflow for carbon diamond.
- A two-state excited-state optimization workflow.
- A stable overlap or orthogonality diagnostic.
- A physically reasonable excitation energy.
- A cost estimate precise enough to decide whether Phase 2 is justified.

## Current Engineering Focus

The backend route has been proven for the carbon-diamond ground-state
benchmark. The FermiNet PBC-HF pretraining route is implemented and validated
for the current carbon-diamond Gamma cc-pVDZ workflow. The current engineering
focus is to reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state
VMC method in code, then test it on concrete periodic materials.

Current route:

```text
DeepSolid/FermiNet carbon-diamond reproduction -> PBC-HF pretraining -> penalty-VMC excited-state reproduction -> material tests
```

The DeepSolid and FermiNet carbon-diamond reproductions have passed. The
FermiNet PBC-HF pretraining milestone has also passed for diamond Gamma
cc-pVDZ, with mixed short-wall-clock conclusions. The next engineering step is
to implement the paper-style excited-state penalty objective and run controlled
two-state periodic NES-VMC probes before broader material tests.

## Non-Goals For The First Version

These are important, but they should not enter the first implementation unless
Phase 1 requires them:

- NiO strong-correlation production calculations.
- MoS2 spin-orbit-coupled dark excitons.
- NV-center embedding.
- A polished general-purpose package.
- Large-scale data generation.

## Research Principles

1. Keep the first target small and falsifiable.
2. Record decisions before expanding the scope.
3. Prefer reproducible configs over hard-coded experiment settings.
4. Separate method development, experiment execution, and analysis.
5. Treat failure modes as publishable scientific information when they are well
   diagnosed.

## Working Cycle

Every assistant or coding-agent work session must start by reading
`AGENTS.md`, then `CURRENT_STATUS.md`, then `ACTIVE_TASK.md`.

Each substantial work session should update:

- `ACTIVE_TASK.md` whenever the exact small-step state changes.
- `records/run_index.md` whenever a new numbered task bundle is created,
  submitted, completed, failed, or analyzed.
- `tasks/TASK_LEDGER.md` whenever a numbered task completes or materially
  changes.
- `PROGRESS.md` as a short rolling current-state snapshot.
- `records/progress/` for dated progress notes.
- `records/decisions/` when a strategic choice is made.
- The relevant config or experiment README when an experiment changes.

## Decision Rule

No Phase 2 system should be started until Phase 1 has produced either:

- A clear go signal, with stable periodic excited-state optimization, or
- A clear no-go report, with diagnosed bottlenecks and an adjusted research
  direction.
