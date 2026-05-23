# Progress: SLURM CPU/GPU Smoke Execution

Date: 2026-05-21

## Summary

Actual SLURM submissions were executed for both CPU and GPU smoke tests.

## CPU Attempt 1

Job:

```text
120610
```

Result:

```text
FAILED 1:0
node036
elapsed 00:00:11
```

Cause:

```text
ModuleNotFoundError: No module named 'DeepSolid'
```

The job used:

```text
SOLIDNES_DEEPSOLID_ROOT=/tmp/deepsolid_backend_survey
```

Compute nodes do not share the login node's `/tmp` checkout.

## Fix

Copied the inspected DeepSolid checkout into a project-local path:

```text
external/deepsolid
```

Updated SLURM job templates to default to:

```text
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
```

`external/` is gitignored.

## CPU Attempt 2

Job:

```text
120615
```

Result:

```text
COMPLETED 0:0
node036
elapsed 00:00:49
```

Log confirmed:

```text
DeepSolid process smoke completed
```

Relevant logs:

```text
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/logs/slurm/solidnes-cpu-smoke_120615.log
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/logs/slurm/solidnes-cpu-smoke_120615.err
```

## GPU Attempt

Job:

```text
120612
```

Result:

```text
FAILED 1:0
gpu001
elapsed 00:00:18
selected_partition: intelgpu80g
gres: gpu:1
```

The SLURM scheduling layer worked: the job was placed on a GPU node.

The runtime environment failed the GPU backend check:

```text
jax=0.4.30
jaxlib=0.4.30
jax_default_backend=cpu
jax_devices=[CpuDevice(id=0)]
JAX did not report a GPU device
```

Interpretation:

```text
solidnes-deepsolid-jax0430-probe is CPU-only.
```

The GPU failure is an environment issue, not a SLURM scheduling issue.

Relevant logs:

```text
tasks/phase1_diamond_c/sto3g/smoke/0002_deepsolid_gpu_device_smoke/logs/slurm/solidnes-gpu-smoke_120612.log
tasks/phase1_diamond_c/sto3g/smoke/0002_deepsolid_gpu_device_smoke/logs/slurm/solidnes-gpu-smoke_120612.err
```

## Next Action

Create a separate CUDA-enabled DeepSolid probe environment before retrying GPU
runtime smoke.

Do not modify the working CPU probe environment in place.
