# FermiNet PBC Excited-State Adapter Source

Date: 2026-05-24, Asia/Shanghai

## Summary

Promoted the FermiNet/JAX PBC excited-state adapter wrapper pattern from the
validation script into reusable SolidNES source:

```text
src/solidnes/excited_states/ferminet_pbc_adapter.py
```

The validation script now calls this module:

```text
scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

## Implementation

The new source module provides:

```text
FermiNetPBCExternalStateAdapter
FermiNetJAXModules
assert_pbc_external_state_config
build_external_state_adapter
init_external_state_params
make_network_from_config
make_tiny_state_samples
wrap_signed_network
wrap_pbc_local_energy
```

The adapter keeps the critical PBC rule explicit:

```text
cfg.system.states == 0
```

SolidNES manages one FermiNet parameter tree per target state externally,
because the upstream FermiNet PBC Hamiltonian rejects `states > 0` and
`state_specific=True`.

## Validation

Default environment checks:

```bash
python -m compileall src/solidnes/excited_states \
  scripts/validation/check_ferminet_pbc_excited_adapter_build.py

PYTHONPATH=src python - <<'PY'
import solidnes.excited_states as es
print('import ok', es.FermiNetPBCExternalStateAdapter.__name__)
PY

PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

Real FermiNet/JAX build-only check:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

Output:

```text
ferminet_pbc_excited_adapter_build: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml
jax_platform: cpu
network_type: ferminet
external_state_params: 2
cfg_system_states: 0
nelec: 12
natoms: 2
walkers_per_state: 1
wavefunction_sign_shape: (2, 2, 1)
wavefunction_logabs_shape: (2, 2, 1)
local_energy_wrapped: yes
local_energy_evaluated: no
```

## Task Bundle Decision

No numbered task bundle was created. This was source consolidation plus local
build-only validation and did not produce durable Slurm, backend log,
checkpoint, result, or task-bundle artifacts.

## Next Step

Connect the reusable real FermiNet adapter to the penalty objective path so a
single call can return wavefunction matrix, overlap diagnostics, state energies,
and penalty terms.
