# Run 0067: FermiNet PBC Paper-Tangent Training Smoke, Walkers 2

Date: 2026-05-25

## Purpose

Rerun the fixed two-state real-local-energy/Laplacian paper-tangent training
smoke with two walkers per state.

Run `0066` proved the 0065 NaN objective issue was fixed, but the default
one-walker smoke is degenerate for seed `47`: the centered score-function
energy tangent is zero and the sampled overlap tangent also gives zero
gradient. A local cheap-local-energy check passed with two walkers per state.

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 2]`
- finite initial and final state-energy vectors with shape `[2]`
- finite initial and final overlap matrices with shape `[2, 2]`
- finite true and gradient penalty objectives for two guarded updates
- finite nonzero gradient and parameter-update norms for two guarded updates
- accepted finite candidate update at each step

The smoke does not require objective decrease; it checks executability of the
real local-energy/Laplacian path with the fixed paper-tangent guarded update.

## Command Shape

The task uses the standard FermiNet GPU submitter with:

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0067_ferminet_pbc_paper_tangent_training_smoke_walkers2
```

## Status

Completed as Slurm job `128677`.

The generated plan selected `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores,
an exclusive allocation, and a 30-minute walltime. The job completed in
`00:05:04` with exit code `0:0`.

Summary:

```text
status: ok
jax_platform: gpu
walkers_per_state: 2
steps: 2
initial_penalty_objective: -11.952571868896484
final_penalty_objective: -11.967228889465332
objective_delta: -0.014657020568847656
step_0_grad_l2_norm: 557.01806640625
step_1_grad_l2_norm: 555.3357543945312
step_0_update_accepted: true
step_1_update_accepted: true
```

Result files:

- `results/validation/real_local_energy_training_smoke_summary.md`
- `results/validation/real_local_energy_training_smoke_summary.json`
