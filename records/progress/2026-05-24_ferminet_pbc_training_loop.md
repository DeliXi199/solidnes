# FermiNet PBC Fixed-Sample Training Loop

Date: 2026-05-24

## Summary

Added a reusable fixed-sample training-loop helper for externally managed
FermiNet PBC penalty states:

```text
src/solidnes/excited_states/ferminet_pbc_training.py
```

The helper keeps the established FermiNet PBC constraint that
`cfg.system.states == 0` and updates one external parameter tree per target
state. It returns structured per-step diagnostics for:

```text
penalty objective
weighted state energy
state energies
off-diagonal squared overlap
overlap penalty
maximum absolute off-diagonal overlap
collapse flag
gradient L2 norm
parameter-update L2 norm
```

No numbered task bundle was created because this was source integration plus
local/build-only validation only. It did not submit Slurm or create new durable
compute-result artifacts.

## Validation

```bash
python -m compileall src/solidnes/excited_states \
  scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py
```

```text
Listing 'src/solidnes/excited_states'...
Compiling 'src/solidnes/excited_states/ferminet_pbc_training.py'...
Compiling 'scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py'...
```

```bash
PYTHONPATH=src python - <<'PY'
import solidnes.excited_states as es
print('import ok', es.run_external_state_penalty_sgd.__name__)
print('diag', es.ExternalStatePenaltyStepDiagnostics.__name__)
PY
```

```text
import ok run_external_state_penalty_sgd
diag ExternalStatePenaltyStepDiagnostics
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

Regression checks:

```text
excited-state penalty objective synthetic checks passed
FermiNet PBC excited-state scaffold synthetic checks passed
ferminet_pbc_penalty_terms: ok
ferminet_pbc_penalty_grad_step: ok
```

## Next Step

Use this fixed-sample helper for a tiny real-local-energy multi-step smoke.
Allocate run `0064` only if that smoke is scheduled or produces durable
Slurm/log/result artifacts.
