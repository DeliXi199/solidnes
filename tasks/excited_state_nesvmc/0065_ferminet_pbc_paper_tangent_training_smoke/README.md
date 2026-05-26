# Run 0065: FermiNet PBC Paper-Tangent Training Smoke

Date: 2026-05-25

## Purpose

Retry the two-state FermiNet PBC real-local-energy multi-step smoke after the
run `0064` failure. This run uses the paper-tangent guarded fixed-sample update
path instead of the previous direct real-local-energy `value_and_grad` SGD path.

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 1]`
- finite initial and final state-energy vectors with shape `[2]`
- finite initial and final overlap matrices with shape `[2, 2]`
- finite scalar penalty objectives for two guarded updates
- finite nonzero gradient and parameter-update norms for two guarded updates
- accepted finite candidate update at each step

The smoke does not require objective decrease; it checks executability of the
real local-energy/Laplacian path with the paper-tangent guarded update.

## Command Shape

The task uses the standard FermiNet GPU submitter with:

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0065_ferminet_pbc_paper_tangent_training_smoke
```

## Status

Failed as Slurm job `128674`.

The generated plan selected `intelgpu80g/gpu001` with `gpu:2`, 96 CPU cores,
an exclusive allocation, and a 30-minute walltime. The job ran for `00:06:15`
and exited `1:0`.

The failure was:

```text
ValueError: penalty_objective_step_0 is not finite: nan
```

See
`results/validation/real_local_energy_training_smoke_failure.md`
for the failure analysis. Do not resubmit this task unchanged.
