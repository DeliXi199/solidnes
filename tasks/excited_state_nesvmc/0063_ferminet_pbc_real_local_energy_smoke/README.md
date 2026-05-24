# Run 0063: FermiNet PBC Real Local-Energy Smoke

Date: 2026-05-24

## Purpose

First scheduled two-state FermiNet PBC excited-state smoke using the real
local-energy/Laplacian path instead of the cheap local-energy stand-in.

## Criterion

The run passes if the backend script completes on a scheduled GPU and writes
finite diagnostics for:

- local-energy matrix with shape `[2, 1]`
- state-energy vector with shape `[2]`
- overlap matrix with shape `[2, 2]`
- scalar penalty objective

## Command Shape

The task uses the standard FermiNet GPU submitter with:

```text
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_smoke.py
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_real_local_energy_smoke.yaml
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke
```

## Status

Completed.

Initial 1 GPU / 8 CPU submission `128435` completed successfully. Per user
request, the task was resubmitted as full-node job `128439` on `intelgpu80g`
using `gpu:2`, 96 CPU cores, and an exclusive allocation.

Final full-node job `128439` completed with exit code `0:0`. JAX reported both
`cuda:0` and `cuda:1`; the validation summary recorded finite diagnostics with
state energies `[-1.520418, -9.994875]` Ha and penalty objective `-4.370518`.
