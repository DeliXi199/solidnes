# FermiNet PBC Penalty-Term Path

Date: 2026-05-24, Asia/Shanghai

## Summary

Connected the reusable real FermiNet PBC external-state adapter to the
penalty-objective path:

```text
src/solidnes/excited_states/ferminet_pbc_adapter.py
```

New entry point:

```text
evaluate_ferminet_pbc_penalty_terms(...)
```

The function takes a `FermiNetPBCExternalStateAdapter`, externally managed
state parameter trees, state-indexed samples, and `penalty_alpha`, then returns:

```text
local_energy
state_energy
psi_ratio
overlap_matrix
offdiag_squared_overlap
overlap_penalty
max_abs_offdiag_overlap
collapse_flag
weighted_state_energy
penalty_alpha
penalty_objective
```

By default it uses the adapter's real PBC local-energy wrapper. Build-only
checks can pass a cheap local-energy stand-in to exercise the full data path
without triggering the expensive PBC Laplacian.

## Validation

Default environment:

```bash
python -m compileall src/solidnes/excited_states \
  scripts/validation/check_ferminet_pbc_penalty_terms.py \
  scripts/validation/check_ferminet_pbc_excited_adapter_build.py

PYTHONPATH=src python - <<'PY'
import solidnes.excited_states as es
print('import ok', es.evaluate_ferminet_pbc_penalty_terms.__name__)
PY

PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

FermiNet/JAX environment:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_terms.py
```

Output:

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

The existing adapter build check also still passed:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

## Task Bundle Decision

No numbered task bundle was created. This step added reusable source and local
build-only validation, but did not produce durable Slurm plans, backend logs,
checkpoints, result files, or task-bundle artifacts.

## Next Step

Add the first differentiable optimization-step scaffold or smoke criterion for
externally managed FermiNet state parameters. Decide separately whether the
first smoke should remain cheap-local-energy only or explicitly schedule a real
PBC local-energy/Laplacian check.
