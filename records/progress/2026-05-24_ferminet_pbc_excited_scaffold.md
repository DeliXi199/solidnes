# 2026-05-24 FermiNet PBC Excited-State Scaffold

## Update

Added a minimal no-compute scaffold that maps the backend-independent
excited-state penalty utilities onto the FermiNet PBC calling pattern.

## Files Added

```text
src/solidnes/excited_states/ferminet_pbc_scaffold.py
scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

## Implemented Surface

The scaffold defines:

- `FermiNetPBCStateSamples`: state-indexed walker data with leading
  `[states, walkers, ...]` axes;
- `evaluate_state_wavefunction_matrix`: evaluate each state wavefunction on
  every state's walker population;
- `wavefunction_ratios_from_matrix`: compute
  `psi_i(r_j) / psi_j(r_j)` ratios for overlap estimation;
- `evaluate_overlap_diagnostics`: bridge the state matrix into the existing
  overlap diagnostics;
- `evaluate_state_energy_estimate`: evaluate each state's local energy on its
  own walker population;
- `evaluate_penalty_scaffold_terms`: combine state energies, overlap
  diagnostics, and penalty-objective terms.

The scaffold does not import FermiNet or JAX. A later backend adapter can
supply vmapped FermiNet network/local-energy callables in the same shape.

## Validation

Ran:

```text
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python -m compileall src/solidnes/excited_states scripts/validation/check_ferminet_pbc_excited_scaffold.py scripts/validation/check_excited_state_penalty_objective.py
```

Results:

```text
FermiNet PBC excited-state scaffold synthetic checks passed
excited-state penalty objective synthetic checks passed
compileall passed
```

## Notes

This still does not consume a run number. It produced reusable source code and
no durable compute artifacts under `tasks/`.

## Next Action

Move from synthetic callables to a build-only FermiNet/JAX adapter check:

1. build a FermiNet PBC config with two externally managed state parameter
   trees, not FermiNet's molecular `cfg.system.states` path;
2. wrap `network.apply` and PBC local energy into the scaffold interface;
3. run an import/config/build-only check in the correct FermiNet environment;
4. create a numbered task bundle only if the build-only check produces durable
   output under `tasks/`.
