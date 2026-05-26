# Run 0064: FermiNet PBC Real Local-Energy Training Smoke

Date: 2026-05-24

## Purpose

First scheduled two-state FermiNet PBC excited-state smoke that runs the
reusable fixed-sample penalty training loop with the real
local-energy/Laplacian path.

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes:

- finite initial and final local-energy matrices with shape `[2, 1]`
- finite initial and final state-energy vectors with shape `[2]`
- finite initial and final overlap matrices with shape `[2, 2]`
- finite scalar penalty objectives for two SGD steps
- finite nonzero gradient and parameter-update norms for two SGD steps

The smoke does not require objective decrease; it checks executability of the
real differentiable path, not variational convergence.

## Command Shape

The task uses the standard FermiNet GPU submitter with:

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_real_local_energy_training_smoke.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke
```

## Status

Failed as Slurm job `128523`.

Dry-run passed. The generated plan queued on `intelgpu80g` with `gpu:2`, 96
CPU cores, an exclusive allocation, and a 30-minute walltime because the
full-node `gpu001` resource was busy at planning time.

The job later started on `intelgpu80g/gpu001` and ran for `00:04:57` with
exit code `1:0`. The scheduler allocation and JAX environment were valid:
the log reported GPU backend devices `cuda:0` and `cuda:1`.

The validation script failed at final-term validation:

```text
ValueError: final_local_energy contains non-finite values: [[nan] [nan]]
```

This means the current direct fixed-sample `value_and_grad` plus plain SGD path
over the real PBC local energy is numerically unsafe. Do not resubmit the same
job unchanged; implement the paper-style loss tangent and clipping/guard logic
first.
