# Implementation Strategy

Last updated: 2026-05-21

## Purpose

This document defines the first implementation strategy for SolidNES.

The immediate goal is not to build a complete ab initio excited-state NQS code.
The immediate goal is to identify the shortest reliable path to the Phase 1
go/no-go test:

```text
Carbon-diamond periodic benchmark, then a controlled two-state extension.
```

## Guiding Decision

Use an adapter-first strategy.

This means SolidNES should first own:

- Project structure.
- Experiment configs.
- Run workflows.
- Diagnostics.
- Progress and decision records.
- Analysis scripts.

SolidNES should initially reuse or adapt existing backends for:

- NQS ansatz implementation.
- Sampling.
- Local energy evaluation.
- Optimizer and preconditioner logic.
- Checkpointing when practical.

## Why Not Rewrite Everything First?

The Phase 1 risk is scientific and numerical, not only software-engineering
risk. Rewriting a full ab initio VMC stack before the first go/no-go test would
delay the key question.

For SolidNES, the expensive and error-prone parts are:

- Periodic all-electron or pseudopotential local energy.
- Electron-electron and electron-ion terms.
- Periodic boundary conditions and finite-size effects.
- FermiNet/Psiformer-style antisymmetric wavefunctions.
- Stable multi-state excited-state optimization.

These should not all be rebuilt before a backend survey is complete.

## Candidate Routes

### Route A: FermiNet-First

Start from FermiNet-style code and add periodic-solid support.

Advantages:

- Closer to molecular excited-state NQS and NES-VMC ideas.
- More natural for multi-state wavefunction optimization.

Risks:

- Periodic solids may require substantial changes.
- Solid-state inputs, finite-size handling, and ECP/PBC behavior may be missing
  or hard to integrate.

### Route B: DeepSolid-First

Start from a solid-state NQS code and add excited-state NES-VMC support.

Advantages:

- Closer to periodic solids.
- Better starting point for diamond and other periodic solids.

Risks:

- Excited-state multi-wavefunction machinery may be a major addition.
- State overlap, state collapse diagnostics, and off-diagonal observables may
  not fit existing assumptions.

### Route C: Adapter-First

Keep SolidNES as the outer research framework and connect to whichever backend
looks most practical after inspection.

Advantages:

- Keeps project control inside SolidNES.
- Allows FermiNet-like and DeepSolid-like routes to be compared.
- Avoids early commitment to a large fork.
- Keeps Phase 1 focused on go/no-go evidence.

Risks:

- Some backends may not be easy to call as libraries.
- The adapter layer may become temporary glue code.
- Core backend edits may still be necessary later.

## Current Recommendation

Use Route C first.

```text
Adapter-first for project control; backend modification when necessary.
```

This is not a decision to avoid core algorithm work. It is a decision to delay
irreversible backend commitment until the minimum viable path is clear.

## Backend Survey Outcome

The first backend survey selected DeepSolid as the first concrete target for
the periodic ground-state smoke path.

This does not replace the adapter-first strategy. The current implementation
stance is:

```text
Adapter-first project structure, DeepSolid-first ground-state smoke path.
```

FermiNet remains the primary reference for NES-VMC and excited-state machinery.

## Lessons From VMC_reproduce

The VMC_reproduce project is useful as a reference because it combines:

- Self-owned experiment configs.
- Self-owned workflow code.
- Self-owned diagnostics and progress notes.
- External libraries for lower-level numerical work.

SolidNES should borrow the pattern, not the exact code layout.

Borrow:

- Configs split by role: experiment, system, model, sampler, train.
- A thin `scripts/run_config.py` entry point later.
- Workflow modules for run orchestration.
- Human-readable progress notes and handoff documents.
- Results excluded from version control.

Do not borrow directly:

- The dual `pre_vmc/` and `vmc/` tree structure.
- The assumption that the whole physics engine should be reimplemented early.
- The moire-specific Hamiltonian and model code.

## First Month Plan

### Week 1: Backend Survey

Deliverables:

- Backend survey notes.
- Initial backend comparison table.
- Decision record for the first implementation route.

Questions:

- Which candidate backend can run a periodic ground-state solid calculation?
- Which candidate backend exposes local energy and sampling cleanly?
- Which backend is easiest to extend to multi-state optimization?
- Which backend already has ECP or pseudopotential support?

### Week 2: Minimal Interface Design

Deliverables:

- Draft interfaces for system, backend, training, and diagnostics.
- No large implementation yet.

Candidate interfaces:

```text
SystemSpec
WavefunctionBackend
SamplerBackend
EnergyEstimator
ExcitedStateObjective
TrainingDiagnostics
```

### Week 3: Ground-State Smoke Path

Deliverables:

- One config-driven ground-state smoke run plan.
- Logging and checkpoint requirements.
- Energy, variance, and acceptance diagnostics.

### Week 4: Two-State NES-VMC Probe Design

Deliverables:

- Two-state config.
- Overlap matrix diagnostic.
- State collapse diagnostic.
- Excitation energy reporting plan.

## Entry Criteria For Algorithm Code

Do not start a full NES-VMC implementation until:

- The backend survey is complete.
- The Phase 1 ground-state path is chosen.
- Config structure is stable.
- Required diagnostics are listed.
- A decision record explains why the chosen backend is acceptable.
