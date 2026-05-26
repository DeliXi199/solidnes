# Run 0069: FermiNet PBC Driver Real Local-Energy Walkers4 Smoke

Date: 2026-05-25

## Purpose

Scale the sampler-integrated FermiNet PBC excited-state driver smoke from run
`0068` from two to four walkers per external state.

This keeps the same two-state driver loop, paper-tangent guarded update,
Metropolis sampling, and real FermiNet PBC local-energy/Laplacian path, but
checks the first nontrivial larger walker matrix shape.

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 4]`
- finite sampler acceptance in `[0, 1]` for each driver iteration
- finite true and gradient penalty objectives for each driver iteration
- finite nonzero gradient and parameter-update norms for each driver iteration
- accepted guarded update for each driver iteration
- successful checkpoint roundtrip

This is still an executability and diagnostics smoke, not a convergence run.

## Command Shape

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_driver_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_driver_real_local_energy_walkers4_smoke.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0069_ferminet_pbc_driver_real_local_energy_walkers4_smoke
```

The backend script reads walker count and sampler settings from the experiment
YAML. `SOLIDNES_BACKEND_ARGS` remains available for explicit overrides.

## Status

Completed.

Job `128752` ran on `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores, and
exclusive allocation. It finished in `00:04:31` with exit code `0:0`.

Summary:

- `jax_platform`: `gpu`
- `local_energy_source`: `real_pbc`
- walkers/state: `4`
- sampler acceptance: `0.7916666865`, `0.75`
- penalty objective: `-2.1405215263 -> -4.2503957748`
- guarded updates accepted: `true`, `true`
- checkpoint roundtrip bytes: `1234734`
