# 2026-05-24 Excited-State Penalty Utilities

## Update

Implemented the first backend-independent SolidNES utilities for the
Szabo-Noe penalty-based excited-state VMC route.

## Files Added

```text
src/solidnes/excited_states/__init__.py
src/solidnes/excited_states/overlap.py
src/solidnes/excited_states/penalty.py
scripts/validation/check_excited_state_penalty_objective.py
```

## Implemented Surface

Overlap utilities:

- estimate overlap matrices from all-state wavefunction ratios;
- symmetrize non-symmetric overlap estimates with the clipped geometric mean;
- compute upper-triangle squared-overlap penalties;
- expose max off-diagonal overlap and collapse diagnostics.

Penalty utilities:

- weighted state-energy aggregation;
- energy-gap overlap-gradient scaling;
- energy-std overlap-gradient scaling;
- max-gap/std scaling;
- scalar penalty-VMC objective terms.

## Validation

Ran the no-compute synthetic check:

```text
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
```

Result:

```text
excited-state penalty objective synthetic checks passed
```

Also ran:

```text
PYTHONPATH=src python -m compileall src/solidnes/excited_states scripts/validation/check_excited_state_penalty_objective.py
```

## Notes

This work did not create a numbered task bundle because it produced no SLURM
plan, backend log, checkpoint, validation result directory, or compute-result
artifact. The next task bundle should be created only when a build-only,
smoke, training, evaluation, or analysis step needs project artifacts under
`tasks/`.

## Next Action

Connect these utilities to a minimal FermiNet PBC excited-state scaffold:

1. define the data/interface shape for two state parameter trees and two walker
   populations;
2. evaluate all state wavefunctions on all state samples for overlap
   diagnostics;
3. bridge PBC local-energy evaluation without relying on FermiNet's molecular
   excited-state path;
4. create a numbered task bundle only when a build/smoke check produces
   durable project artifacts.
