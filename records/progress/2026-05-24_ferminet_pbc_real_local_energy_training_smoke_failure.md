# FermiNet PBC Real Local-Energy Training Smoke Failure

Date: 2026-05-24, Asia/Shanghai

## Summary

Run `0064` attempted the first scheduled two-state FermiNet PBC fixed-sample
penalty training-loop smoke using the real local-energy/Laplacian path.

```text
Task root: tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke/
Job ID: 128523
Partition/node: intelgpu80g/gpu001
Resources: gpu:2, cpu:96, exclusive
Elapsed: 00:04:57
Exit code: 1:0
```

The job reached the CUDA environment correctly. JAX reported the GPU backend
and both visible devices, `cuda:0` and `cuda:1`.

## Result

The run failed in the validation script while checking the final terms:

```text
ValueError: final_local_energy contains non-finite values: [[nan] [nan]]
```

The normal success summary was not written because validation raised before the
summary write. A failure summary was added at:

```text
tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke/results/validation/real_local_energy_training_smoke_failure.md
tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke/results/validation/real_local_energy_training_smoke_failure.json
```

## Interpretation

This is not a Slurm or resource-allocation failure. The plan selected
`intelgpu80g`, requested both A100 80GB GPUs, 96 CPU cores, and an exclusive
allocation, and the job started correctly.

The failure is method-side: the current helper differentiates the scalar
penalty objective through the real FermiNet PBC local-energy path with
`jax.value_and_grad`, then applies a plain fixed-sample SGD update. Run `0064`
shows that this direct path can produce non-finite final local energies even
with learning rate `1e-8` and one walker per state.

## Next Step

Do not resubmit `0064` unchanged.

The next source step is to implement a paper-faithful penalty-VMC training
tangent before another real-local-energy multi-step smoke:

- ordered excited-state update with lower-state stop-gradient behavior
- clipped wavefunction-ratio and overlap estimator behavior
- local-energy clipping or finite local-energy guards
- automatic penalty scaling from gap/std diagnostics instead of a fixed raw
  penalty alpha
- finite-gradient and finite-update guards before parameters are committed

This next step is source work. It should not allocate a new numbered task
folder until another SLURM smoke or durable validation artifact is created.
