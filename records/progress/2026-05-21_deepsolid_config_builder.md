# Progress: DeepSolid Config Builder Added

Date: 2026-05-21

## What Was Added

Added:

```text
scripts/backends/build_deepsolid_config.py
configs/sampler/metropolis_deepsolid_smoke.yaml
configs/train/ground_state_deepsolid_smoke.yaml
configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
src/solidnes/backends/deepsolid_configs/diamond_smoke.py
```

## What Passed

The original DeepSolid diamond config route passed config construction:

```text
configs/experiment/diamond_c_deepsolid_ground_smoke.yaml
```

Summary:

```text
nelectron: 12
nelec: (6, 6)
basis: ccpvdz
batch_size: 8
optimizer: adam
iterations: 0
hidden_dims: ((64, 16), (64, 16))
determinants: 4
```

The lower-cost SolidNES runtime smoke route also passed config construction:

```text
configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Summary:

```text
nelectron: 12
nelec: (6, 6)
basis: sto-3g
batch_size: 8
optimizer: adam
iterations: 0
hidden_dims: ((64, 16), (64, 16))
determinants: 4
```

## Interpretation

SolidNES now has a working adapter-style config layer:

```text
SolidNES YAML -> DeepSolid config object
```

The next step is to decide whether to run a zero-iteration DeepSolid process
smoke on the lower-cost `sto-3g` route. That run will test more than config
construction: Hartree-Fock initialization, network initialization, MCMC setup,
and local-energy evaluation.

## Caveat

The `sto-3g` runtime smoke is not a physics benchmark. It exists only to test
runtime compatibility.
