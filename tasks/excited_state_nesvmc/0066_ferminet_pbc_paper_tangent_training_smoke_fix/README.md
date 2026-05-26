# Run 0066: FermiNet PBC Paper-Tangent Training Smoke Fix

Date: 2026-05-25

## Purpose

Verify the fix for run `0065`, where the first per-step paper-tangent
`penalty_objective` diagnostic became `nan`.

This run keeps the same two-state real-local-energy/Laplacian fixed-sample
training smoke shape, but uses the updated training path:

- true penalty terms are evaluated outside `value_and_grad`
- paper-tangent gradients reuse precomputed local-energy values
- the surrogate zero-primal correction is forward-safe
- per-step diagnostics record both true `penalty_objective` and
  `gradient_objective`

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 1]`
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
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_fix.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0066_ferminet_pbc_paper_tangent_training_smoke_fix
```

## Status

Failed as Slurm job `128675`.

The generated plan selected `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores,
an exclusive allocation, and a 30-minute walltime. The job ran for `00:05:01`
and exited `1:0`.

The failure was:

```text
ValueError: grad_l2_norm_step_0 must be positive, got 0.0
```

See
`results/validation/real_local_energy_training_smoke_failure.md`
for the failure analysis. Do not resubmit this task unchanged; the next smoke
should use at least two walkers per state.
