# DeepSolid Environment Check

Date: 2026-05-21

## Summary

Checked whether existing local Python environments can run DeepSolid.

## Findings

Default environment:

```text
Python 3.12.3
jax MISSING
jaxlib MISSING
pyscf MISSING
ml_collections MISSING
optax MISSING
chex MISSING
```

Existing `crystalformer` environment:

```text
Python 3.10.20
jax 0.6.2
jaxlib 0.6.2
optax 0.2.6
chex 0.1.90
pyscf MISSING
ml_collections MISSING
```

Top-level imports of cloned repositories work when added to `PYTHONPATH`, but
DeepSolid config code fails without `ml_collections` and `pyscf`.

## Decision

Use an isolated legacy DeepSolid environment and run carbon diamond first.

See:

```text
records/decisions/0004_deepsolid_environment_and_smoke_target.md
```
