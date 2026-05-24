# Production Readiness Decision

Created: `2026-05-24`

## Decision

Use `jax_pbc_gto` for the next diamond Gamma cc-pVDZ pretraining pilot, with
`pretrain_jax_pbc_image_cutoff: 3`.

This is not yet a blanket replacement for all PBC systems. It is validated for
the current diamond primitive Gamma workflow.

## Evidence

- `sto-3g` AO/MO validation passed at `image_cutoff=2`.
- `sto-3g` GPU benchmark showed `5.39x` total pretrain step speedup and
  `19.19x` target-eval speedup versus PySCF target.
- `ccpvdz` AO validation passed at `image_cutoff=3`:
  AO max abs `1.12e-9`, AO RMS abs `8.00e-11`.
- `ccpvdz` occupied MO validation passed at `image_cutoff=3`:
  MO max abs `8.51e-10`, MO RMS abs `1.00e-10`.
- `ccpvdz` GPU benchmark 0050 completed:
  mean step `0.023191s`, target eval `0.015405s`.
- PySCF cc-pVDZ benchmark 0047 measured:
  mean step `0.058384s`, target eval `0.051142s`.

## Guardrails

- Continue requiring `pretrain_jax_pbc_image_cutoff >= 3` for `ccpvdz`.
- Keep PySCF target backend available for cross-checks.
- Re-run AO/MO validation before using `jax_pbc_gto` for non-Gamma twists,
  different cells, pseudopotentials, or basis sets beyond `sto-3g`/`ccpvdz`.

## Next Task

Create a longer pilot that combines:

```yaml
pretrain_target_backend: jax_pbc_gto
pretrain_basis: ccpvdz
pretrain_jax_pbc_image_cutoff: 3
pretrain_iterations: 1000
optimizer: kfac
```

The pilot should compare downstream early-training energy and pmove against the
current FermiNet cc-pVDZ baseline, not only pretrain target timing.
