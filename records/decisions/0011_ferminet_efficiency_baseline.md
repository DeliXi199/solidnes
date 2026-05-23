# Decision 0011: Use FermiNet For The Efficiency-First Diamond Baseline

Date: 2026-05-23

## Decision

SolidNES will add FermiNet as the next primary backend route for carbon-diamond
PBC energy reproduction and acceleration benchmarking.

DeepSolid remains useful as a reference-result and comparison backend, but the
efficiency-first path should start from FermiNet because upstream FermiNet
already contains PBC infrastructure, FOLX Forward Laplacian plumbing, KFAC,
PsiFormer support, and active JAX-oriented infrastructure.

## Immediate Scope

- Reproduce all-electron primitive-cell diamond Gamma-point ground-state energy.
- Use latest available JAX from PyPI at the time of this decision:
  `jax==0.10.1` / `jaxlib==0.10.1`.
- Start with a small FermiNet PBC smoke config and a KFAC baseline config.
- Enable easy low-risk acceleration defaults first: FOLX Forward Laplacian,
  KFAC, FP64 off, GPU memory preallocation, and PBC-specific feature/envelope
  functions.

## Deferred Scope

- PBC excited states.
- PBC pseudopotentials/ECP.
- Shared twist training.
- Mixed precision beyond the current FP32/TF32-oriented speed profile.
- Multi-node sharding.

## Rationale

DeepSolid has already produced useful carbon-diamond reference trajectories, but
its old JAX/KFAC path makes further acceleration work expensive. FermiNet gives
the project a cleaner baseline for measuring time-to-energy and makes later
Forward Laplacian, PsiFormer, twist sharing, and distributed training work more
credible.

