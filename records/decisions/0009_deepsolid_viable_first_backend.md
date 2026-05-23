# Decision 0009: Treat DeepSolid As The First Backend

Date: 2026-05-21

## Decision

SolidNES will continue with DeepSolid as the first backend for the Phase 1
periodic ground-state path.

This decision is limited to the smoke-test and adapter-building level. It does
not claim that DeepSolid is already validated for production training, KFAC, or
NES-VMC.

## Rationale

DeepSolid passed the required Phase 0C smoke layers:

- Import/config smoke in an isolated JAX 0.4.30 CPU probe environment.
- Carbon diamond config construction.
- Zero-iteration carbon diamond runtime smoke on CPU.
- SLURM CPU smoke on job `120615`.
- CUDA JAX probe environment creation.
- Zero-iteration carbon diamond runtime smoke on GPU job `120634`.
- One-iteration Adam carbon diamond runtime smoke on GPU job `120655`.

The successful zero-iteration GPU job reported:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
DeepSolid process smoke completed
```

The earlier GPU failure on job `120612` was an environment issue: the probe
environment was CPU-only. It was not a DeepSolid or SLURM scheduling failure.

The first one-step Adam attempt on job `120654` reached training statistics but
failed during historical DeepSolid checkpoint saving. A smoke-only object-array
checkpoint save shim resolved that compatibility issue, and retry job `120655`
completed.

## Consequences

- The script-level config builder has been refactored into
  `src/solidnes/backends/deepsolid_adapter.py`.
- The adapter should expose reusable objects before NES-VMC work starts:
  config, system metadata, output paths, compatibility status, and later
  network/local-energy/MCMC handles.
- The CPU and CUDA probe environments must remain separate:
  `solidnes-deepsolid-jax0430-probe` for CPU smoke and
  `solidnes-deepsolid-jax0430-cuda12-probe` for GPU smoke.
- The smoke-only compatibility shim remains a temporary bridge, not production
  evidence for KFAC training.

## Revisit Trigger

Revisit this decision if later reusable-internals work exposes a deeper
DeepSolid/JAX incompatibility.
