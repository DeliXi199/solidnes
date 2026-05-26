# Run 0065 Failure: FermiNet PBC Paper-Tangent Training Smoke

Date: 2026-05-25

## Status

```text
status: failed
job_id: 128674
partition: intelgpu80g
node: gpu001
elapsed: 00:06:15
exit_code: 1:0
resources: gpu:2, cpu:96, exclusive
```

## Command

```text
backend_script: scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke.yaml
task_root: tasks/excited_state_nesvmc/0065_ferminet_pbc_paper_tangent_training_smoke
gradient_mode: paper_tangent
```

## Failure

```text
ValueError: penalty_objective_step_0 is not finite: nan
```

The Slurm allocation and FermiNet/JAX GPU environment were valid. The job
started on `gpu001`, JAX reported both CUDA devices, and the failure occurred
inside the validation script after the fixed-sample training helper returned.

## Interpretation

Run 0064 failed after a direct real-local-energy update because the final local
energy became non-finite. Run 0065 moved to the guarded paper-tangent update
path, but the first per-step `penalty_objective` diagnostic was still `nan`.

The validation script checks initial and final penalty terms before inspecting
per-step diagnostics. Therefore this failure points to the `value_and_grad`
paper-tangent scalar returned by the gradient path, not to the separately
evaluated true penalty terms. The likely source is the zero-value tangent
construction:

```text
stop_gradient(true_objective) + surrogate - stop_gradient(surrogate)
```

If `surrogate` is non-finite in the forward pass, the intended zero-valued
surrogate correction becomes `nan - nan`, so the diagnostic objective becomes
`nan` even when the true objective terms are finite.

## Next Action

Make the paper-tangent objective forward-safe and record the true pre-update
penalty objective in step diagnostics. Then rerun local validation and submit a
new numbered GPU smoke instead of resubmitting 0065 unchanged.
