# Run 0066 Failure: FermiNet PBC Paper-Tangent Training Smoke Fix

Date: 2026-05-25

## Status

```text
status: failed
job_id: 128675
partition: intelgpu80g
node: gpu001
elapsed: 00:05:01
exit_code: 1:0
resources: gpu:2, cpu:96, exclusive
```

## Command

```text
backend_script: scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_fix.yaml
task_root: tasks/excited_state_nesvmc/0066_ferminet_pbc_paper_tangent_training_smoke_fix
seed: 47
walkers_per_state: 1
gradient_mode: paper_tangent
```

## Failure

```text
ValueError: grad_l2_norm_step_0 must be positive, got 0.0
```

## Interpretation

The 0065 `penalty_objective_step_0 = nan` failure was resolved: run 0066
reached the gradient-norm assertion instead. The new failure is a smoke-design
issue for this seed and sample size. With one walker per state, the
score-function energy tangent is centered over a single sample and contributes
zero. For seed `47`, the overlap tangent also has zero gradient, so the guarded
fixed-sample update has `grad_l2_norm = 0.0`.

This was reproduced locally with the cheap-local-energy path:

```text
seed=47, walkers=1, steps=2 -> grad_norm_step_0 = 0.0
seed=47, walkers=2, steps=2 -> passed, grad_l2_norm_step_0 ~= 31.85
```

## Next Action

Do not resubmit 0066 unchanged. Allocate a new task using at least two walkers
per state so the score-function tangent is meaningful and the smoke can test a
real parameter update.
