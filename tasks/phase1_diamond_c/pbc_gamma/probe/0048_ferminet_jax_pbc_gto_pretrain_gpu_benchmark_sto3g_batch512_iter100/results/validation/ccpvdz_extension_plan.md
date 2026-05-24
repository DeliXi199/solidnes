# cc-pVDZ Extension Plan

Created: `2026-05-24`

The `jax_pbc_gto` target should remain gated to `sto-3g` until these checks pass.

## Correctness Tasks

1. Add a focused PySCF comparison for individual d-shell AO columns.
2. Confirm PySCF spherical harmonic ordering for d shells, not only s/p shells.
3. Validate cc-pVDZ contracted basis values shell-by-shell against
   `cell.eval_gto("PBCGTOval_sph", ...)`.
4. Repeat Gamma AO/MO comparison for diamond cc-pVDZ with `image_cutoff=1,2,3`.
5. Pick the smallest image cutoff whose AO/MO error is below the target tolerance.
6. Only then allow:

```yaml
pretrain_target_backend: jax_pbc_gto
pretrain_basis: ccpvdz
```

## Current Guardrail

The adapter currently rejects `pretrain_target_backend: jax_pbc_gto` unless
`pretrain_basis: sto-3g`. This prevents accidentally running an unvalidated
cc-pVDZ JAX target in production pretraining.
