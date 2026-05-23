# Decision 0005: Add A Newer JAX DeepSolid Probe Environment

Date: 2026-05-21

## Decision

Keep the exact legacy DeepSolid environment file as documentation, but add a
second probe environment based on Python 3.9 and JAX 0.4.30.

Status after execution: accepted for the next smoke-test layer.

New probe file:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
```

## Rationale

The exact DeepSolid pin requires `jaxlib==0.1.75`, but that package was not
available from the current package source during the first environment creation
attempt. The resulting environment is partial and unusable.

Using a newer JAX probe is the fastest way to answer the next practical
question:

```text
Does DeepSolid's config/build path fail immediately on newer JAX,
or is the old pin stricter than necessary for the smoke-test layer?
```

The probe is intentionally scoped to import and config construction first. It
does not claim that production training is valid on this stack.

## Consequences

- `solidnes-deepsolid-legacy` should be considered broken until rebuilt.
- The next runnable environment is `solidnes-deepsolid-jax0430-probe`.
- Import and diamond config construction passed in the probe environment.
- Training compatibility remains unproven and must be tested with a short
  backend smoke run before any real optimization.

## Revisit Trigger

Revisit after the probe environment either:

- passes `scripts/backends/check_deepsolid_environment.py`, or
- fails with concrete JAX/PySCF/DeepSolid API errors.
