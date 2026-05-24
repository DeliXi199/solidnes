# FermiNet PBC Excited-State Adapter Build Check

Date: 2026-05-24, Asia/Shanghai

## Summary

Added the first real FermiNet/JAX build-only check for the SolidNES
penalty-state scaffold:

```text
scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

The check deliberately does not set FermiNet's molecular excited-state config:

```text
cfg.system.states == 0
```

This matters because upstream FermiNet's PBC Hamiltonian raises
`NotImplementedError` when `states > 0` or `state_specific` is enabled.

## What The Check Proves

The script builds an existing carbon-diamond Gamma FermiNet PBC config from:

```text
configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml
```

It then:

1. Constructs the real FermiNet PBC network from the built config.
2. Initializes two externally managed state parameter trees.
3. Wraps `network.apply` into the scaffold's batched signed-network interface.
4. Evaluates the wavefunction matrix on tiny synthetic state-indexed walkers.
5. Constructs a PBC local-energy wrapper with `states=0` and
   `state_specific=False`.

By default the script does not evaluate local energy, because that would trigger
the expensive Laplacian path and is beyond the intended build-only scope.

## Command Run

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

Syntax check:

```bash
python -m compileall scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

## Task Bundle Decision

No numbered task bundle was created. This was a source-level validation script
and a local build-only check; it did not produce durable SLURM plans, backend
logs, checkpoints, validation summaries, or result artifacts under `tasks/`.

## Next Step

Promote the adapter wrapper pattern into reusable SolidNES source, then connect
the real FermiNet wavefunction matrix/state-energy path to the penalty objective
before creating any smoke or Slurm task bundle.
