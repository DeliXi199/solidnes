# Decision 0006: Use Two Diamond Smoke Layers

Date: 2026-05-21

## Decision

Use two separate carbon-diamond smoke paths:

```text
diamond_c_deepsolid_ground_smoke
diamond_c_deepsolid_runtime_smoke
```

Status after execution: accepted.

## Rationale

The original DeepSolid diamond config uses `ccpvdz`, which is closer to the
published repository pattern but may be too expensive for quick CPU runtime
checks.

The SolidNES-owned `diamond_smoke.py` config uses `sto-3g` and looser PySCF
precision. This makes it better suited for checking whether the runtime path
still works under the newer JAX 0.4.30 probe environment.

## Consequences

- Use `diamond_c_deepsolid_ground_smoke` for compatibility with DeepSolid's
  original config style.
- Use `diamond_c_deepsolid_runtime_smoke` for tiny code-path execution tests.
- Do not report `sto-3g` smoke results as physically meaningful.

## Revisit Trigger

Revisit after a zero-iteration or one-iteration DeepSolid process smoke has
either passed or failed with a concrete compatibility error.

Current result:

```text
zero-iteration DeepSolid process smoke passed
```
