# FermiNet PBC Driver Controlled12 Run

Date: 2026-05-25

## Summary

Added a production backend runner for the sampler-integrated FermiNet PBC
external-state excited-state driver, then ran the first controlled trajectory
beyond smoke scale.

## Runner

New script:

```text
scripts/backends/run_ferminet_pbc_excited_driver.py
```

Capabilities:

- reads run settings from experiment YAML diagnostics
- supports cheap local-energy debugging and real PBC local-energy runs
- runs in checkpoint-sized segments
- saves driver checkpoints containing state params, samples, sampler key, and
  resume metadata
- resumes from a saved driver checkpoint
- writes JSON and Markdown trajectory summaries

The local cheap-local-energy resume check passed: one run wrote
`driver_iter_000001.pkl`, and a second run resumed from it to iteration 2.

## Scheduled Run

Task:

```text
tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4
```

Job `128758` completed on `intelgpu80g/gpu001` in `00:14:28` with exit `0:0`.
The job used both A100 80GB GPUs and 96 CPU cores.

Run settings:

```text
states: 2
walkers_per_state: 4
iterations: 12
checkpoint_interval: 4
local_energy_source: real_pbc
learning_rate: 1.0e-8
```

Results:

```text
initial_penalty_objective: 56.4663162231
final_penalty_objective: -13.3468618393
final_state_energy: [-17.5776462555, -11.4508419037]
final_overlap_offdiag: 0.4831940234
```

All 12 guarded updates were accepted with finite nonzero gradient/update norms.
Checkpoints were written at iterations 4, 8, and 12.

Trajectory analysis artifacts:

```text
ferminet_pbc_driver_trajectory.csv
ferminet_pbc_driver_trajectory_analysis.md
```

Analysis snapshot:

```text
objective_min: -24.4335079193
acceptance_mean: 0.8090277811
grad_norm_max: 9789.92871094
final_overlap_offdiag: 0.4831940234
```

## Next Work

The next useful acceleration is to run a controlled trajectory that is actually
closer to production sampling: increase walkers per state and sampler steps,
then add a small analysis script that plots or tabulates objective, state
energy, overlap, acceptance, gradient norm, and checkpoint cadence.
