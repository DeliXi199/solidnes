# DeepSolid Environment Plan

Last updated: 2026-05-21

## Purpose

This note records the current DeepSolid runtime status and the next safe steps
for making a backend smoke test runnable.

## Current Local Environment Check

Default Python:

```text
/data/home/yihaoxu/Python/anaconda3/bin/python
Python 3.12.3
```

Currently missing from the default environment:

```text
jax
jaxlib
pyscf
ml_collections
optax
chex
```

Existing conda environment `crystalformer` has:

```text
Python 3.10.20
jax 0.6.2
jaxlib 0.6.2
optax 0.2.6
chex 0.1.90
```

But it is missing:

```text
pyscf
ml_collections
```

and it does not match DeepSolid's pinned legacy stack.

## DeepSolid Dependency Mismatch

The inspected DeepSolid `setup.py` pins:

```text
jax==0.2.26
jaxlib==0.1.75
optax==0.0.9
chex==0.1.5
scipy==1.9.3
h5py==3.2.1
```

This is not compatible with the current default Python 3.12 workflow. DeepSolid
should be isolated in its own environment.

## Current Import Status

With `PYTHONPATH=/tmp/deepsolid_backend_survey`, top-level import succeeds:

```text
import DeepSolid
```

But useful config imports fail in the default environment:

```text
from DeepSolid import base_config
ModuleNotFoundError: No module named 'ml_collections'

from DeepSolid.config import diamond
ModuleNotFoundError: No module named 'pyscf'
```

## Recommended Environment Route

Create a separate legacy environment:

```text
solidnes-deepsolid-legacy
```

Use the environment scaffold:

```text
configs/env/deepsolid_legacy_cpu.yml
```

Do not install DeepSolid into the base environment.

## Environment Creation Attempt 1

Status: failed.

Command target:

```text
configs/env/deepsolid_legacy_cpu.yml
```

Observed result:

```text
solidnes-deepsolid-legacy was partially created.
Python 3.9.23 is available inside the environment.
```

But the pip stage failed because the current package source could not provide:

```text
jaxlib==0.1.75
```

The partially created environment is not usable:

```text
jax False
jaxlib False
pyscf False
ml_collections False
optax False
chex False
```

`h5py` also fails to import inside the partial environment with an HDF5 symbol
error, so this environment should be treated as broken until it is removed and
rebuilt.

## Revised Environment Strategy

Keep `configs/env/deepsolid_legacy_cpu.yml` as a record of DeepSolid's published
legacy dependency intent, but do not assume it is reproducible from the current
package source.

Add a second environment route:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
```

This probe route intentionally uses a newer available JAX line. It is not the
final production environment. Its job is narrower:

1. Check whether DeepSolid config construction works with a newer JAX stack.
2. Identify concrete API breakages, if any.
3. Decide whether a small compatibility patch is cheaper than recovering the
   exact 2021-era JAX stack.

If this probe fails at import/config time, the next options are:

1. Recover the historical JAX wheel from an official/archive source.
2. Patch DeepSolid against a newer JAX API.
3. Switch the first backend smoke test to a FermiNet/PBC ground-state path while
   keeping DeepSolid as a reference implementation.

## Environment Creation Attempt 2

Status: passed for import/config smoke.

Environment:

```text
solidnes-deepsolid-jax0430-probe
```

Created from:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
```

Validated package versions:

```text
Python 3.9.23
jax 0.4.30
jaxlib 0.4.30
pyscf 2.13.0
optax 0.2.2
chex 0.1.86
scipy 1.13.1
numpy 1.26.4
```

DeepSolid checks passed:

```text
DeepSolid.base_config: OK batch_size=100
DeepSolid diamond config: OK nelectron=12 nelec=(6, 6)
```

Interpretation:

The newer JAX probe is good enough for the next smoke-test layer: config
construction and small backend integration checks. It is not yet validated for
full DeepSolid training.

## Runtime Smoke Result

Status: passed for zero-iteration runtime smoke.

Command:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=/tmp/deepsolid_backend_survey:solidnes/src \
  python scripts/backends/run_deepsolid_process_smoke.py
```

Final smoke settings:

```text
experiment: diamond_c_deepsolid_runtime_smoke
basis: sto-3g
precision: 1e-8
optimizer: adam
iterations: 0
use_x64: true
```

This run completed DeepSolid's process path and produced:

```text
tasks/phase1_diamond_c/sto3g/smoke/0001_deepsolid_runtime_smoke/results/checkpoints/train_stats.csv
```

Notes:

- The compatibility shim is smoke-only.
- The result is not a physics benchmark.
- KFAC training remains unvalidated on JAX 0.4.30.

## SLURM Runtime Status

CPU SLURM smoke passed after using a project-local DeepSolid checkout:

```text
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
job 120615
COMPLETED 0:0
```

GPU SLURM smoke was submitted and placed on `gpu001`, but failed before running
DeepSolid because the current probe environment is CPU-only:

```text
jax_default_backend=cpu
jax_devices=[CpuDevice(id=0)]
JAX did not report a GPU device
```

Immediate conclusion at that point:

```text
Use solidnes-deepsolid-jax0430-probe for CPU smoke.
Build and validate a separate CUDA probe environment before retrying GPU smoke.
```

## CUDA Probe Environment And GPU Smoke

Status: passed for zero-iteration GPU runtime smoke.

Environment:

```text
solidnes-deepsolid-jax0430-cuda12-probe
```

Created from:

```text
configs/env/deepsolid_py39_jax0430_cuda12_probe.yml
```

Validated import/config smoke on the login node:

```text
DeepSolid.base_config: OK batch_size=100
DeepSolid diamond config: OK nelectron=12 nelec=(6, 6)
```

The login node has no visible GPU, so direct `jax.default_backend()` checks fail
there with `No visible GPU devices`. The real GPU validation was performed
inside SLURM.

GPU SLURM smoke:

```text
job 120634
partition gpu4090_128
node gpu40903
state COMPLETED 0:0
elapsed 00:00:36
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
DeepSolid process smoke completed
```

Conclusion:

```text
Use solidnes-deepsolid-jax0430-probe for CPU smoke.
Use solidnes-deepsolid-jax0430-cuda12-probe for GPU smoke.
```

## One-Step Adam GPU Smoke

Status: passed after adding a smoke-only checkpoint save shim.

First attempt:

```text
job 120654
FAILED 1:0
partition gpu4090_128
node gpu40903
```

The failed attempt reached training statistics output but failed in
`DeepSolid.checkpoint.save` because modern JAX/Optax pytrees were passed
directly to `np.savez`.

Compatibility shim:

```text
src/solidnes/backends/deepsolid_compat.py
patch_checkpoint_save_for_smoke()
```

Successful retry:

```text
job 120655
COMPLETED 0:0
partition gpu4090_128
node gpu40903
elapsed 00:00:33
DeepSolid process smoke completed
```

Output:

```text
tasks/phase1_diamond_c/sto3g/smoke/0004_deepsolid_one_step_adam_checkpoint_shim_smoke/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/sto3g/smoke/0004_deepsolid_one_step_adam_checkpoint_shim_smoke/results/checkpoints/qmcjax_ckpt_000000.npz
```

## Environment Check Script

After an environment exists, run:

```bash
PYTHONPATH=/path/to/DeepSolid \
python scripts/backends/check_deepsolid_environment.py
```

The script checks:

- Required Python modules.
- `DeepSolid.base_config.default()`.
- `DeepSolid.config.diamond.get_config("C,C,3.57,1,ccpvdz")`.

## First Smoke Target

Use carbon diamond as the first periodic solid smoke target.

Reason:

- DeepSolid already has a diamond config.
- It tests the periodic solid backend path without overcommitting compute.

The first smoke config is:

```text
configs/experiment/diamond_c_deepsolid_ground_smoke.yaml
```
