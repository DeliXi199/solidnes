# Decision 0007: Allow Smoke-Only DeepSolid Compatibility Shims

Date: 2026-05-21

## Decision

Use small compatibility shims to run DeepSolid smoke tests on the JAX 0.4.30
probe environment.

Shim file:

```text
src/solidnes/backends/deepsolid_compat.py
```

## Rationale

Recovering the exact historical `jax==0.2.26` / `jaxlib==0.1.75` environment was
blocked by package availability. The JAX 0.4.30 probe environment installs
cleanly, but DeepSolid's legacy KFAC support imports removed JAX symbols.

For zero-iteration smoke tests, KFAC tags are not needed. The shim restores
minimal legacy aliases and neutralizes KFAC tag primitives only for smoke runs.
For one-step Adam smoke, the shim also redirects DeepSolid checkpoint saving to
object-array `np.savez` so modern JAX/Optax pytrees can be written without
editing the external DeepSolid checkout.

## Constraints

The shim is allowed for:

```text
import/config smoke
zero-iteration runtime smoke
one-step Adam runtime smoke
adapter interface development
```

The shim is not sufficient for:

```text
production KFAC training
physical benchmark results
published calculations
```

## Consequences

Before production runs, choose one:

1. Recover the exact legacy DeepSolid environment.
2. Port DeepSolid/KFAC properly to modern JAX.
3. Use a different backend for training and keep DeepSolid as a reference.
