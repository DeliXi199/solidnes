# Progress: DeepSolid Adapter And One-Step Adam Smoke

Date: 2026-05-21

## Summary

SolidNES moved from script-level DeepSolid config construction to a source-level
adapter module, then used that adapter path to run a one-iteration Adam GPU
smoke.

## Adapter Work

Added:

```text
src/solidnes/backends/deepsolid_adapter.py
```

The adapter now owns:

- SolidNES experiment YAML loading.
- Role-split config resolution.
- DeepSolid template import and config construction.
- Model/sampler/training config application.
- Output path construction.
- Runtime compatibility metadata.
- Summary formatting for CLI wrappers.

Updated scripts:

```text
scripts/backends/build_deepsolid_config.py
scripts/backends/run_deepsolid_process_smoke.py
```

The scripts are now thin wrappers around the adapter.

## One-Step Adam Config

Added:

```text
configs/train/ground_state_deepsolid_one_step_adam.yaml
configs/experiment/diamond_c_deepsolid_one_step_adam_smoke.yaml
```

This target keeps the same tiny carbon-diamond runtime system as the
zero-iteration smoke, but sets:

```text
optimizer: adam
iterations: 1
batch_size: 8
use_x64: true
```

## First One-Step Attempt

Job:

```text
120654
```

Result:

```text
FAILED 1:0
partition gpu4090_128
node gpu40903
elapsed 00:00:32
```

Interpretation:

The run reached training statistics output and wrote step `0` to
`train_stats.csv`, but failed during `DeepSolid.checkpoint.save`.

Error:

```text
ValueError: setting an array element with a sequence
```

Cause:

DeepSolid's historical checkpoint writer passes modern JAX/Optax pytrees
directly to `np.savez`, which NumPy tries to coerce into homogeneous arrays.

## Checkpoint Shim

Added smoke-only checkpoint saving compatibility:

```text
src/solidnes/backends/deepsolid_compat.py
```

Function:

```text
patch_checkpoint_save_for_smoke()
```

The shim saves checkpoint components as object arrays and does not modify the
external DeepSolid checkout.

## Successful One-Step Retry

Job:

```text
120655
```

Result:

```text
COMPLETED 0:0
partition gpu4090_128
node gpu40903
elapsed 00:00:33
```

Runtime log confirmed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
DeepSolid process smoke completed
```

Outputs:

```text
tasks/phase1_diamond_c/sto3g/smoke/0004_deepsolid_one_step_adam_checkpoint_shim_smoke/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/sto3g/smoke/0004_deepsolid_one_step_adam_checkpoint_shim_smoke/results/checkpoints/qmcjax_ckpt_000000.npz
```

`train_stats.csv` contains one row:

```text
step=0
energy=-24.258438174974998
variance=96.05468583373442
pmove=1.0
```

## Interpretation

DeepSolid is now validated for:

- CPU zero-iteration runtime smoke.
- CUDA GPU zero-iteration runtime smoke.
- CUDA GPU one-iteration Adam smoke.

This still does not validate production KFAC training, physical benchmark
quality, or NES-VMC extensions.

## Next Action

Start the carbon-diamond backend-interface step:

```text
Expose reusable DeepSolid backend objects for adapter-level method work.
```
