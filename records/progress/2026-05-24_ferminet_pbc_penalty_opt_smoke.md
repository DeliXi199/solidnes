# FermiNet PBC Penalty Multi-Step Optimization Smoke

Date: 2026-05-24

## Summary

Added a build-only multi-step optimization smoke for the FermiNet PBC
external-state penalty objective:

```text
scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py
```

The check initializes two external FermiNet PBC state parameter trees, uses the
real adapter/scaffold path, substitutes a cheap local-energy stand-in, and runs
three deterministic SGD updates. This validates that the current external-state
penalty objective can produce finite gradients across consecutive updates and
move the objective in the expected direction before introducing the expensive
real PBC local-energy/Laplacian path.

No numbered task bundle was created because this was a local build-only source
validation with no Slurm submission and no durable checkpoint, log, result, or
task-bundle artifact.

## Validation

```bash
python -m compileall src/solidnes/excited_states \
  scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py \
  scripts/validation/check_ferminet_pbc_penalty_grad_step.py
```

```text
Listing 'src/solidnes/excited_states'...
Compiling 'scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py'...
```

```bash
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
```

```text
excited-state penalty objective synthetic checks passed
```

```bash
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

```text
FermiNet PBC excited-state scaffold synthetic checks passed
```

```bash
PYTHONPATH=src python - <<'PY'
import solidnes.excited_states as es
print('import ok', es.apply_external_state_sgd_step.__name__)
PY
```

```text
import ok apply_external_state_sgd_step
```

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py
```

```text
ferminet_pbc_penalty_opt_smoke: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml
jax_platform: cpu
cfg_system_states: 0
external_state_params: 2
local_energy_source: cheap
steps: 3
initial_penalty_objective: 12.67329216
final_penalty_objective: 12.6731863022
objective_delta: -0.000105857849121
step_0: objective=12.673289299 grad_l2_norm=64.2644500732 param_delta_l2_norm=6.41282952074e-07
step_1: objective=12.6732578278 grad_l2_norm=64.2560195923 param_delta_l2_norm=6.41244696453e-07
step_2: objective=12.6732196808 grad_l2_norm=64.2434463501 param_delta_l2_norm=6.41062399609e-07
```

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_grad_step.py
```

```text
ferminet_pbc_penalty_grad_step: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml
jax_platform: cpu
cfg_system_states: 0
external_state_params: 2
local_energy_source: cheap
penalty_objective: 11.9213724136
updated_penalty_objective: 11.7314357758
grad_l2_norm: 59.3455619812
param_delta_l2_norm: 0.00593455787748
```

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_terms.py
```

```text
ferminet_pbc_penalty_terms: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml
jax_platform: cpu
cfg_system_states: 0
external_state_params: 2
local_energy_source: cheap
state_energy_shape: (2,)
overlap_matrix_shape: (2, 2)
psi_ratio_shape: (2, 2, 2)
local_energy_shape: (2, 2)
penalty_objective_shape: ()
collapse_flag_shape: ()
```

## Notes

The first draft used a `1e-4` toy SGD learning rate and produced a NaN gradient
on the second step. The default was reduced to `1e-8`, which is appropriate for
this smoke because the goal is to prove repeated differentiable updates in the
real adapter path, not to stress-test the optimizer.

The next implementation step is to define a controlled carbon-diamond Gamma
two-state smoke using the real PBC local-energy/Laplacian path. Allocate run
`0063` only if that step is scheduled or produces durable task artifacts.
