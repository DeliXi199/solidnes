# Progress: DeepSolid Runtime Smoke Passed

Date: 2026-05-21

## Command

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=/tmp/deepsolid_backend_survey:solidnes/src \
  python scripts/backends/run_deepsolid_process_smoke.py --timeout-seconds 300
```

## Result

Passed.

Final output included:

```text
experiment: diamond_c_deepsolid_runtime_smoke
starting DeepSolid process smoke
params initialization seed: 888
DeepSolid process smoke completed
```

The run created:

```text
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/results/checkpoints/train_stats.csv
```

Because `iterations: 0`, the CSV contains only the header.

## Issues Found And Resolved

The runtime smoke exposed several compatibility issues:

1. DeepSolid's legacy KFAC code expected `jax.xla.translations`.
2. It also expected `jax.tree_util.tree_multimap`.
3. It used the removed type alias `jnp.DeviceArray`.
4. `use_x64: false` caused DeepSolid's Ewald consistency assertion to fail.
5. `optimizer: none` hit a DeepSolid double-`pmap` path during initial energy
   evaluation.

Resolution:

```text
src/solidnes/backends/deepsolid_compat.py
configs/train/ground_state_deepsolid_smoke.yaml
```

The final smoke uses:

```text
optimizer: adam
iterations: 0
use_x64: true
```

## Interpretation

DeepSolid is viable as the first backend for SolidNES at the smoke-test level.

This does not yet validate production training, KFAC, or
excited-state/NES-VMC extensions.
