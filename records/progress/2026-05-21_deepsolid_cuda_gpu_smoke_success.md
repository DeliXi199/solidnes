# Progress: DeepSolid CUDA GPU Smoke Passed

Date: 2026-05-21

## Summary

A CUDA-enabled DeepSolid probe environment was created and used successfully for
the GPU zero-iteration runtime smoke.

## Environment

Environment:

```text
solidnes-deepsolid-jax0430-cuda12-probe
```

Config:

```text
configs/env/deepsolid_py39_jax0430_cuda12_probe.yml
```

Key package versions:

```text
Python 3.9.23
jax 0.4.30
jaxlib 0.4.30
jax-cuda12-plugin 0.4.30
jax-cuda12-pjrt 0.4.30
pyscf 2.13.0
optax 0.2.2
chex 0.1.86
numpy 1.26.4
scipy 1.13.1
```

Login-node import/config smoke passed:

```text
DeepSolid.base_config: OK batch_size=100
DeepSolid diamond config: OK nelectron=12 nelec=(6, 6)
```

The login node has no visible GPU, so direct CUDA backend initialization is
expected to fail there. GPU backend validation was done through SLURM.

## GPU Job

Job:

```text
120634
```

Result:

```text
COMPLETED 0:0
partition gpu4090_128
node gpu40903
elapsed 00:00:36
```

The job used:

```text
SOLIDNES_CONDA_ENV=solidnes-deepsolid-jax0430-cuda12-probe
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Runtime log confirmed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
DeepSolid process smoke completed
```

Relevant logs:

```text
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/logs/slurm/solidnes-gpu-cuda12-smoke_120634.log
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/logs/slurm/solidnes-gpu-cuda12-smoke_120634.err
```

## Interpretation

DeepSolid is viable as the first SolidNES backend at the zero-iteration
CPU/GPU smoke-test level.

This does not yet validate one-iteration optimization, production training,
KFAC, or excited-state/NES-VMC extensions.

## Next Action

Refactor the script-level DeepSolid builder into:

```text
src/solidnes/backends/deepsolid_adapter.py
```

Then decide whether to run a one-iteration Adam smoke before moving toward the
carbon-diamond backend-interface milestone.
