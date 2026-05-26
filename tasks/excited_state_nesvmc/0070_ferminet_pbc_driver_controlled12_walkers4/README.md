# Run 0070: FermiNet PBC Driver Controlled12 Walkers4

Date: 2026-05-25

## Purpose

Run the first controlled sampler-integrated FermiNet PBC excited-state
trajectory beyond smoke scale.

This advances beyond runs `0068` and `0069` by using the production backend
runner instead of the validation-only smoke script. The run targets 12 driver
iterations, four walkers per external state, real FermiNet PBC
local-energy/Laplacian diagnostics, and periodic driver checkpoints.

## Criterion

The run passes if the backend runner completes on scheduled GPUs and writes:

- `ferminet_pbc_driver_run_summary.json`
- `ferminet_pbc_driver_run_summary.md`
- 12 finite sampler/update diagnostic rows
- finite final local-energy matrix with shape `[2, 4]`
- accepted guarded update for every iteration
- checkpoint files at iterations `4`, `8`, and `12`

This is still a controlled diagnostic trajectory, not a converged production
science result.

## Command Shape

```text
SOLIDNES_BACKEND_SCRIPT=scripts/backends/run_ferminet_pbc_excited_driver.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_driver_controlled12_walkers4.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4
```

The runner reads iteration count, walker count, sampler settings, and
checkpoint interval from the experiment YAML. `SOLIDNES_BACKEND_ARGS` remains
available for explicit overrides.

## Status

Completed.

Job `128758` ran on `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores, and
exclusive allocation. It finished in `00:14:28` with exit code `0:0`.

Summary:

- `jax_platform`: `gpu`
- `local_energy_source`: `real_pbc`
- walkers/state: `4`
- completed iterations: `12`
- checkpoint interval: `4`
- checkpoint files: `driver_iter_000004.pkl`, `driver_iter_000008.pkl`,
  `driver_iter_000012.pkl`
- penalty objective: `56.4663162231 -> -13.3468618393`
- final state energy: `[-17.5776462555, -11.4508419037]`
- final overlap off diagonal: `0.4831940234`
- all guarded updates accepted
- trajectory analysis: objective min `-24.4335079193`, acceptance mean
  `0.8090277811`, max grad norm `9789.92871094`
