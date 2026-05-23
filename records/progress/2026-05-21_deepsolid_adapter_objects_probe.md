# Progress: DeepSolid Adapter Objects Probe

Date: 2026-05-21

## Summary

SolidNES can now initialize and call the DeepSolid ground-state runtime pieces
outside DeepSolid's monolithic training loop.

The new adapter surface is:

```text
src/solidnes/backends/deepsolid_adapter.py
DeepSolidGroundStateObjects
initialize_deepsolid_ground_state(...)
```

The probe command is:

```text
scripts/backends/probe_deepsolid_adapter_objects.py
```

## What The Probe Exposes

- DeepSolid `ConfigDict`.
- PySCF simulation cell.
- Hartree-Fock object.
- Network objects for matrix, logdet, and slogdet evaluation.
- Batched network apply functions.
- Replicated parameters.
- Replicated walker batch.
- Pmap-wrapped local-energy loss.
- Pmap-wrapped MCMC transition.
- Replicated MCMC width and sharded RNG key.

This is the object boundary needed before SolidNES can build a NES-VMC layer
without driving DeepSolid through the full `process.process(...)` loop.

## CPU Result

Command:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/probe_deepsolid_adapter_objects.py \
  configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml \
  --timeout-seconds 300
```

Key output:

```text
experiment: diamond_c_deepsolid_runtime_smoke
basis: sto-3g
pseudo: None
nelectron: 12
nelec: (6, 6)
runtime_objects: initialized
num_devices: 1
local_batch_size: 8
data_shape: (1, 8, 36)
param_leaf_count: 12
first_param_shape: (1, 16)
energy: -26.0833736842
variance: 61.3415623746
pmove: 1
DeepSolid adapter object probe completed
```

## CUDA GPU Result

The GPU probe was submitted with:

```bash
SOLIDNES_GPU_PARTITIONS=h200,h800,h20,gpu4090_128 \
SOLIDNES_BACKEND_SCRIPT=scripts/backends/probe_deepsolid_adapter_objects.py \
SOLIDNES_JOB_NAME=solidnes-gpu-adapter-probe \
SOLIDNES_PLAN_JSON=tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/outputs/slurm_plans/deepsolid_gpu_adapter_probe_plan.json \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

Final status:

```text
job 120686
node gpu40904
partition gpu4090_128
state COMPLETED
exit code 0:0
elapsed 00:00:59
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
```

Key output:

```text
runtime_objects: initialized
data_shape: (1, 8, 36)
param_leaf_count: 12
energy: -28.834877001
variance: 100.485069341
mcmc_data_shape: (1, 8, 36)
pmove: 1
DeepSolid adapter object probe completed
```

The CPU and GPU scalar values are smoke diagnostics only. The current target is
interface reachability and finite execution, not a converged or transferable
physics result.

## SLURM Wrapper Change

The CPU and GPU DeepSolid SLURM templates now accept:

```text
SOLIDNES_BACKEND_SCRIPT
```

The default remains:

```text
scripts/backends/run_deepsolid_process_smoke.py
```

This lets the same scheduling policy run process smoke, one-step smoke, and
adapter-object probes without duplicating SLURM templates.

## Next

Build a SolidNES-native backend state interface on top of these DeepSolid
objects:

```text
log_psi(params, walkers)
local_energy(params, walkers)
mcmc_step(state)
energy_diagnostics(state)
```

Then add the first small carbon-diamond two-state/NES objective scaffold. That
will test method plumbing on the retained benchmark route.
