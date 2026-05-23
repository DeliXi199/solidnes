# Progress: DeepSolid Environment Creation Attempt

Date: 2026-05-21

## What Was Attempted

Tried to create the isolated DeepSolid legacy environment from:

```text
configs/env/deepsolid_legacy_cpu.yml
```

The target environment name was:

```text
solidnes-deepsolid-legacy
```

## Result

The environment was only partially created.

Confirmed:

```text
Python 3.9.23
/data/home/yihaoxu/Python/anaconda3/envs/solidnes-deepsolid-legacy/bin/python
```

Missing:

```text
jax
jaxlib
pyscf
ml_collections
optax
chex
```

The install failed at the pip dependency stage because `jaxlib==0.1.75` was not
available from the current package source for this environment.

`h5py` also fails to import in the partial environment:

```text
undefined symbol: H5Pset_fapl_ros3
```

## Interpretation

The public DeepSolid dependency pin is historically useful, but it is not
currently reproducible through the default package route.

This is an environment reproducibility issue, not yet evidence that DeepSolid
cannot be used for the SolidNES smoke test.

## Follow-Up

Added a newer JAX probe environment:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
```

The next execution step is to create that probe environment and run:

```bash
PYTHONPATH=/tmp/deepsolid_backend_survey \
python scripts/backends/check_deepsolid_environment.py
```
