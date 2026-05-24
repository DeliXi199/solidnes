# FermiNet PBC Penalty Gradient Step

Date: 2026-05-24, Asia/Shanghai

## Summary

Added the first differentiable optimization-step scaffold for externally
managed FermiNet PBC state parameter trees.

New reusable helpers:

```text
src/solidnes/excited_states/ferminet_pbc_adapter.py
```

```text
ferminet_pbc_penalty_objective(...)
value_and_grad_ferminet_pbc_penalty_objective(...)
apply_external_state_sgd_step(...)
```

The objective still keeps the critical PBC rule:

```text
cfg.system.states == 0
```

SolidNES owns the state dimension through separate FermiNet parameter trees.

## Validation

Default environment:

```bash
python -m compileall src/solidnes/excited_states \
  scripts/validation/check_ferminet_pbc_penalty_grad_step.py \
  scripts/validation/check_ferminet_pbc_penalty_terms.py

PYTHONPATH=src python - <<'PY'
import solidnes.excited_states as es
print('import ok', es.value_and_grad_ferminet_pbc_penalty_objective.__name__)
PY

PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

FermiNet/JAX environment:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_grad_step.py
```

Output:

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

The check confirms:

```text
1. The scalar penalty objective is finite.
2. Gradients over external state parameter trees are finite and nonzero.
3. A simple SGD update changes the parameter trees.
4. The updated objective can be evaluated again.
```

## Task Bundle Decision

No numbered task bundle was created. This step added reusable source and a
local build-only gradient validation; it did not produce durable Slurm plans,
backend logs, checkpoints, result files, or task-bundle artifacts.

## Next Step

Define the first carbon-diamond Gamma two-state smoke criterion. The decision
is whether to keep the first smoke cheap-local-energy only or to schedule a real
PBC local-energy/Laplacian check under the next numbered task bundle.
