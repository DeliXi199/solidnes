# Decision 0004: DeepSolid Environment And First Smoke Target

Date: 2026-05-21

## Decision

Do not use the base Python environment for DeepSolid.

Create an isolated DeepSolid legacy environment before running backend smoke
tests.

Use carbon diamond as the first real-solid DeepSolid smoke target.

## Rationale

The default Python environment is Python 3.12 and lacks the required DeepSolid
runtime dependencies. DeepSolid's public setup is pinned to an older stack,
including `jax==0.2.26` and `jaxlib==0.1.75`, so installing it into the base
environment would be risky.

Carbon diamond is a better first backend smoke target because it:

- Uses the existing DeepSolid diamond config pattern.
- Still tests a real periodic solid.

## Consequences

- Add `configs/env/deepsolid_legacy_cpu.yml` as the environment scaffold.
- Add `scripts/backends/check_deepsolid_environment.py`.
- Add `configs/experiment/diamond_c_deepsolid_ground_smoke.yaml`.
- Keep the first backend target narrow until carbon-diamond runs are stable.

## Revisit Trigger

Revisit this decision after:

- The DeepSolid legacy environment is created and checked.
- The carbon diamond config import/build smoke test succeeds.
- A second target system is selected.
