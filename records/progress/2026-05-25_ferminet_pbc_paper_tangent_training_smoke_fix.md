# FermiNet PBC Paper-Tangent Training Smoke Fix

Date: 2026-05-25

## Summary

The real FermiNet PBC local-energy/Laplian fixed-sample training smoke now
passes with the paper-tangent guarded update path.

The final passing task is:

```text
run: 0067
job_id: 128677
partition: intelgpu80g
node: gpu001
resources: gpu:2, cpu:96, exclusive
elapsed: 00:05:04
exit_code: 0:0
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml
summary: tasks/excited_state_nesvmc/0067_ferminet_pbc_paper_tangent_training_smoke_walkers2/results/validation/real_local_energy_training_smoke_summary.md
```

## Failures Resolved

Run `0065` failed at:

```text
ValueError: penalty_objective_step_0 is not finite: nan
```

Root cause: the paper-tangent objective computed the true real-local-energy
terms inside `value_and_grad` and used a `surrogate - stop_gradient(surrogate)`
zero-primal correction that can still produce a NaN forward value when the
surrogate is non-finite.

Fix:

- evaluate true penalty terms outside `value_and_grad`
- reuse precomputed local-energy values in the paper-tangent gradient objective
- record true pre-update `penalty_objective` separately from
  `gradient_objective`
- make the surrogate zero-primal correction forward-safe
- add a cheap-local-energy regression checking no extra local-energy calls
  happen inside `value_and_grad`

Run `0066` then failed at:

```text
ValueError: grad_l2_norm_step_0 must be positive, got 0.0
```

Root cause: the one-walker smoke is degenerate for seed `47`; the centered
score-function energy tangent is zero with one sample, and the sampled overlap
tangent also gave zero gradient. A local cheap-local-energy check reproduced
the failure with `walkers=1` and passed with `walkers=2`.

Fix:

- use `walkers_per_state: 2` for the scheduled real-local-energy training
  smoke
- update the real-local-energy training smoke script so YAML diagnostics can
  provide defaults such as `walkers_per_state`

## Passing Result

Run `0067` completed successfully:

```text
status: ok
jax_platform: gpu
jax_devices: cuda:0, cuda:1
walkers_per_state: 2
steps: 2
gradient_mode: paper_tangent
local_energy_source: real_pbc
elapsed_seconds: 279.276906
initial_penalty_objective: -11.952571868896484
final_penalty_objective: -11.967228889465332
objective_delta: -0.014657020568847656
```

Per-step guards:

```text
step 0: grad_l2_norm=557.01806640625, param_delta_l2_norm=5.602138116955757e-06, update_accepted=True
step 1: grad_l2_norm=555.3357543945312, param_delta_l2_norm=5.5834761951700784e-06, update_accepted=True
```

Both steps had finite true and gradient objectives, finite gradients, finite
updates, finite candidate terms, accepted updates, and no collapse flag.

## Next Step

The fixed-sample real-local-energy training path is now smoke-validated. The
next engineering step is to move from fixed tiny samples to the production
excited-state driver: sampler integration, optimizer state/checkpointing, and
longer controlled runs with stable diagnostics.
