# Decision 0002: Initial Implementation Strategy

Date: 2026-05-21

## Decision

Use an adapter-first implementation strategy for Phase 1.

SolidNES should first own the research framework, configs, workflows, progress
records, and diagnostics. It should not immediately become a full fork of
FermiNet, DeepSolid, or any other backend.

## Rationale

The first scientific milestone is a controlled periodic-solid NQS benchmark
that can support a later excited-state extension. That milestone does not
require a polished general-purpose package. It requires a reliable path to:

- Periodic ground-state VMC.
- Two-state excited-state optimization.
- Orthogonality or overlap diagnostics.
- Excitation energy and variance reporting.
- A credible cost estimate.

The adapter-first route keeps the project flexible while the backend survey is
still open.

## Consequences

- Do not add large algorithm code before backend survey.
- Keep `src/solidnes/` minimal for now.
- Split configs by role so they can later target different backends.
- Use experiment folders and progress records as the project memory.
- Treat FermiNet-first and DeepSolid-first as candidate backend routes, not
  immediate commitments.

## Revisit Trigger

Revisit this decision when one of the following happens:

- A backend is selected for the ground-state periodic-solid smoke test.
- A backend cannot be called cleanly and must be forked.
- NES-VMC requires changes too deep for the adapter layer.
- Phase 1 produces a go/no-go result.
