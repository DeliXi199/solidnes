# DeepSolid Run Config Notes

## Purpose

This note records the intended first backend route for SolidNES:

```text
SolidNES config -> DeepSolid external backend -> periodic ground-state smoke test
```

## Candidate DeepSolid Command Shape

The inspected DeepSolid README uses commands of this form:

```bash
deepsolid --config=PATH/TO/DeepSolid/config/diamond.py:C,C,3.57,1,ccpvdz --config.batch_size TBD
```

## Immediate Questions

- Which Python/JAX environment can run DeepSolid's pinned dependency stack?
- Which carbon-diamond config should be used for the next smoke or validation
  run?
- How should DeepSolid's output be mirrored into the numbered task bundle?

## Current Environment Finding

The exact legacy environment failed because `jaxlib==0.1.75` was not available
from the current package source. A newer probe environment now passes import and
diamond config construction:

```text
solidnes-deepsolid-jax0430-probe
```

Use:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
scripts/backends/check_deepsolid_environment.py
scripts/backends/build_deepsolid_config.py
scripts/backends/run_deepsolid_process_smoke.py
```

First smoke target:

```text
configs/experiment/diamond_c_deepsolid_ground_smoke.yaml
```

Lower-cost runtime smoke target:

```text
configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
src/solidnes/backends/deepsolid_configs/diamond_smoke.py
```

This runtime smoke uses `sto-3g` and a looser PySCF precision. It is only meant
to test code compatibility and should not be used as a physics benchmark.

## JAX Compatibility Note

The first runtime smoke failed at import time because DeepSolid's legacy KFAC
code expects:

```text
jax.xla.translations
```

which is absent from JAX 0.4.30. SolidNES therefore provides a smoke-only shim:

```text
src/solidnes/backends/deepsolid_compat.py
```

The shim restores the legacy aliases needed for import and neutralizes KFAC tag
primitives for smoke runs that are not using KFAC. It is valid for
zero-iteration runtime smoke only, not for production KFAC training.

Current successful runtime smoke uses:

```text
optimizer: adam
iterations: 0
use_x64: true
```

Reason:

- `use_x64: true` is needed to pass the strict Ewald consistency assertion.
- `optimizer: adam` avoids DeepSolid's `optimizer: none` double-`pmap` initial
  energy path while still doing zero training steps.

## Do Not Do Yet

- Do not start NES-VMC implementation before the ground-state backend path has
  passed at least one tiny runtime smoke.
