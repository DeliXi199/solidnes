# Run 0068: FermiNet PBC Driver Real Local-Energy Smoke

Date: 2026-05-25

## Purpose

Run the first scheduled GPU smoke for the sampler-integrated FermiNet PBC
excited-state driver.

This advances beyond run `0067`, which validated the fixed-sample
paper-tangent real-local-energy training helper. Run `0068` adds the driver
pieces needed by production NES-VMC:

- per-state Metropolis sampling
- paper-tangent guarded parameter updates
- persistent diagnostics
- checkpoint save/load roundtrip
- real FermiNet PBC local-energy/Laplacian path

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 2]`
- finite sampler acceptance in `[0, 1]` for each driver iteration
- finite true and gradient penalty objectives for each driver iteration
- finite nonzero gradient and parameter-update norms for each driver iteration
- accepted guarded update for each driver iteration
- successful checkpoint roundtrip

This is still an executability and diagnostics smoke, not a convergence run.

## Command Shape

The task uses the standard FermiNet GPU submitter with:

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_driver_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_driver_real_local_energy_smoke.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0068_ferminet_pbc_driver_real_local_energy_smoke
```

The backend script reads `local_energy_source`, `iterations`, walker count, and
sampler settings from the experiment YAML. `SOLIDNES_BACKEND_ARGS` remains
available for explicit overrides.

## Status

Completed.

Job `128751` ran on `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores, and
exclusive allocation. It finished in `00:05:44` with exit code `0:0`.

Summary:

- `jax_platform`: `gpu`
- `local_energy_source`: `real_pbc`
- walkers/state: `2`
- sampler acceptance: `0.75`, `0.75`
- penalty objective: `-1.3069219589 -> 21.3078460693`
- guarded updates accepted: `true`, `true`
- checkpoint roundtrip bytes: `1233826`
