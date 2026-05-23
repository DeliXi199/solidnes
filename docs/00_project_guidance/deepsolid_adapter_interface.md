# DeepSolid Adapter Interface

Last updated: 2026-05-21

## Purpose

This document defines the minimal interface SolidNES needs from DeepSolid before
implementing periodic NES-VMC.

The adapter boundary is:

```text
SolidNES experiment YAML -> DeepSolid ConfigDict -> backend smoke or run
```

## Current Status

Passed:

```text
environment import/config smoke
diamond config construction smoke
zero-iteration diamond runtime smoke
zero-iteration diamond CUDA GPU runtime smoke
one-iteration diamond Adam CUDA GPU smoke
direct diamond adapter-object probe on CPU and CUDA GPU
```

Current runnable environments:

```text
CPU: solidnes-deepsolid-jax0430-probe
GPU: solidnes-deepsolid-jax0430-cuda12-probe
```

Current runtime smoke:

```text
configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

## Adapter Inputs

The adapter should consume:

- Experiment config: backend, template module, input string, output paths.
- System config: target material and cell-level metadata.
- Model config: DeepSolid detnet size and determinants.
- Sampler config: MCMC burn-in, steps, proposal width.
- Train config: optimizer, iterations, precision, logging.

## Adapter Outputs

At minimum, the adapter should expose:

- Built DeepSolid `ConfigDict`.
- Electron count and spin tuple.
- Basis and pseudopotential status.
- Output/checkpoint path.
- Runtime compatibility status.

It now also exposes:

- Initialized network apply functions.
- Parameters.
- Walker batch.
- Local energy function.
- MCMC transition function.
- Ground-state energy diagnostics.

## Current Commands

Environment check:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/check_deepsolid_environment.py
```

Config build:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/build_deepsolid_config.py \
  configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Zero-iteration runtime smoke:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/run_deepsolid_process_smoke.py
```

CUDA GPU SLURM smoke:

```bash
SOLIDNES_GPU_PARTITIONS=h200,amdgpu80g,amdgpu40g,intelgpu80g,h20 \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

One-step Adam GPU smoke:

```bash
SOLIDNES_GPU_PARTITIONS=h200,amdgpu80g,amdgpu40g,intelgpu80g,h20 \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_deepsolid_one_step_adam_smoke.yaml \
SOLIDNES_JOB_NAME=solidnes-gpu-one-step-adam \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

Direct adapter-object probe:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/probe_deepsolid_adapter_objects.py \
  configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Direct adapter-object GPU probe:

```bash
SOLIDNES_GPU_PARTITIONS=h200,amdgpu80g,amdgpu40g,intelgpu80g,h20 \
SOLIDNES_BACKEND_SCRIPT=scripts/backends/probe_deepsolid_adapter_objects.py \
SOLIDNES_JOB_NAME=solidnes-gpu-adapter-probe \
SOLIDNES_PLAN_JSON=tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/outputs/slurm_plans/deepsolid_gpu_adapter_probe_plan.json \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

## Rules

- Treat `src/solidnes/backends/deepsolid_compat.py` as smoke-only.
- Do not use `sto-3g` runtime smoke as a physics benchmark.
- Do not treat the direct adapter-object probe as a physics benchmark; it still
  uses the small carbon `sto-3g` smoke setup.

## Current Interface Step

Completed:

```text
src/solidnes/backends/deepsolid_adapter.py
```

The scripts are now thin command-line wrappers around that module.

Latest adapter step:

```text
DeepSolidGroundStateObjects exposes initialized network apply functions,
parameters, walker batches, local energy, and MCMC transition functions.
CPU and CUDA GPU probes have passed.
```

Next adapter step:

```text
Wrap these DeepSolid objects behind a SolidNES-native backend state interface
for NES-VMC scaffolding.
```
