# FermiNet PBC Real Local-Energy Smoke

Date: 2026-05-24

## Summary

Run `0063` passed the first scheduled two-state FermiNet PBC real
local-energy/Laplacian smoke for the excited-state penalty objective path.

The task bundle is:

```text
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/
```

The smoke uses two externally managed FermiNet PBC state parameter trees while
keeping `cfg.system.states == 0`, evaluates the real PBC local-energy wrapper,
and checks finite diagnostics for state energies, overlap, wavefunction ratios,
and the scalar penalty objective.

## Submission

Initial submission:

```text
Job: 128435
Resources: 1 GPU, 8 CPU cores
Result: completed successfully before the cancellation request took effect
Elapsed: 00:02:50
ExitCode: 0:0
```

Per user request, the task was resubmitted using all resources on the selected
`intelgpu80g` node:

```text
Job: 128439
Partition: intelgpu80g
Node: gpu001
Resources: gpu:2, 96 CPU cores, exclusive allocation
Elapsed: 00:02:05
ExitCode: 0:0
```

The full-node dry-run plan was written to:

```text
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/outputs/slurm_plans/fullnode_plan.json
```

The final full-node logs are:

```text
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/logs/slurm/solidnes-nes-real-le-full_128439.log
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/logs/slurm/solidnes-nes-real-le-full_128439.err
```

## Result

The full-node job reported:

```text
jax_platform: gpu
jax_devices: cuda:0, cuda:1
cfg_system_states: 0
external_state_params: 2
walkers_per_state: 1
local_energy_source: real_pbc
elapsed_seconds: 99.370287
local_energy_shape: (2, 1)
state_energy_shape: (2,)
overlap_matrix_shape: (2, 2)
psi_ratio_shape: (2, 2, 1)
penalty_objective_shape: ()
state_energy: [-1.5204181671142578, -9.994874954223633]
penalty_objective: -4.370517730712891
```

The validation files are:

```text
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/results/validation/real_local_energy_smoke_summary.json
tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/results/validation/real_local_energy_smoke_summary.md
```

## Interpretation

This proves that the SolidNES FermiNet PBC external-state penalty path can call
the real FermiNet PBC local-energy/Laplacian wrapper on scheduled GPU resources
for two separately initialized state parameter trees. The output is only a
shape/finite smoke on one walker per state; it is not a physical excited-state
energy claim.

The next step is reusable multi-step training-loop integration around this
real local-energy objective, with per-step state-energy, overlap, off-diagonal
penalty, and collapse diagnostics.
