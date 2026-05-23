# Progress: DeepSolid JAX 0.4.30 Probe Passed

Date: 2026-05-21

## What Was Done

Created the probe environment:

```text
solidnes-deepsolid-jax0430-probe
```

from:

```text
configs/env/deepsolid_py39_jax0430_probe.yml
```

Then ran:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=/tmp/deepsolid_backend_survey \
  python scripts/backends/check_deepsolid_environment.py
```

## Result

The check passed.

Confirmed:

```text
Python 3.9.23
jax 0.4.30
jaxlib 0.4.30
pyscf 2.13.0
ml_collections unknown
optax 0.2.2
chex 0.1.86
scipy 1.13.1
numpy 1.26.4
DeepSolid.base_config: OK batch_size=100
DeepSolid diamond config: OK nelectron=12 nelec=(6, 6)
```

## Interpretation

The project no longer needs to block on exact recovery of the historical
DeepSolid dependency stack.

The JAX 0.4.30 probe environment is sufficient for the next layer:

1. Build a carbon diamond backend smoke script.
2. Confirm DeepSolid can construct the model/training config.
3. Try a very short CPU run only after the config-level smoke passes.

## Caveat

Passing this check does not prove DeepSolid training is compatible with JAX
0.4.30. It only proves import and config construction.
