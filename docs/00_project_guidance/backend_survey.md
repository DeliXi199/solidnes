# Backend Survey

Last updated: 2026-05-21

## Purpose

This survey identifies the first backend route for the Phase 1 ground-state
smoke test:

```text
Carbon-diamond periodic ground-state NQS smoke path.
```

The survey also records what will be needed later for periodic NES-VMC.

## Summary Recommendation

Use DeepSolid as the first backend target for the periodic ground-state smoke
path, but keep the SolidNES project adapter-first.

Short version:

```text
Ground-state periodic solid smoke test: DeepSolid first.
Periodic excited-state/NES-VMC method development: port ideas from FermiNet.
Adapter/interface design: borrow patterns from NetObs.
```

This is not a decision to make SolidNES a DeepSolid fork. It is a decision to
use DeepSolid as the first concrete backend to test the solid-state ground-state
workflow.

## Candidate 1: FermiNet

Repository:

```text
https://github.com/google-deepmind/ferminet
```

Inspected commit:

```text
0ce3411 Merge pull request #110 from APerezFadon:main
commit date: 2026-05-18
```

### Relevant Capabilities

- Actively maintained JAX FermiNet/Psiformer code.
- Supports molecular ground-state VMC.
- Supports molecular excited states via NES-VMC and ensemble overlap penalty.
- Supports pseudopotentials for molecular calculations.
- Contains a `ferminet/pbc/` module with periodic feature layers, periodic
  envelopes, and Ewald local-energy code.
- Contains a homogeneous electron gas PBC example.

### Important Limitation

The current PBC local-energy implementation explicitly rejects excited states:

```text
if states > 0 or state_specific:
  raise NotImplementedError('Excited states not implemented with PBC.')
```

It also rejects pseudopotentials in PBC:

```text
if pp_symbols:
  raise NotImplementedError('Pseudopotentials not implemented with PBC.')
```

### Implication For SolidNES

FermiNet is the best reference for NES-VMC machinery, state handling, and
excited-state diagnostics. It is not the lowest-friction first backend for a
real-solid ground-state smoke run.

FermiNet should be treated as:

- The primary source for NES-VMC implementation ideas.
- A possible future backend after PBC + excited-state integration work.
- Not the first ground-state solid backend.

## Candidate 2: DeepSolid

Repository:

```text
https://github.com/bytedance/DeepSolid
```

Inspected commit:

```text
812a2b8 Merge pull request #6 from AllanChain/fix-requirements
commit date: 2024-12-10
```

### Relevant Capabilities

- Built for periodic solid-state neural-network VMC.
- Developed on top of FermiNet and PyQMC ideas.
- Uses PySCF PBC cells.
- Provides real-solid config examples:
  - diamond
  - graphene
  - rock salt
  - hydrogen chain
  - POSCAR input
- Supports twist through `cfg.network.twist`.
- Includes periodic input features, Ewald terms, MCMC, pretraining, KFAC, and
  checkpointing.
- `DeepSolid/config/diamond.py` directly supports the retained carbon-diamond
  smoke and benchmark path.

### Important Limitations

- No excited-state or NES-VMC support is present in the public code.
- Dependency stack is old:

```text
jax==0.2.26
jaxlib==0.1.75
optax==0.0.9
chex==0.1.5
```

- The public version does not expose a clean modern package interface.
- Pseudopotential/ECP handling is outside the retained carbon-diamond benchmark
  scope.

### Implication For SolidNES

DeepSolid is the best first backend for the periodic ground-state workflow,
especially for testing:

- PySCF PBC cell setup.
- Solid-state periodic features.
- Ewald local energy.
- Periodic MCMC.
- Twist field plumbing.
- Solid-state checkpoint and metrics behavior.

It is not sufficient by itself for Phase 1 NES-VMC.

## Candidate 3: NetObs

Repository:

```text
https://github.com/bytedance/netobs
```

Inspected commit:

```text
35a0413 test against latest FermiNet
commit date: 2025-04-16
```

### Relevant Capabilities

- Provides adaptor abstractions for NN-VMC observables.
- Has built-in adaptors for:
  - FermiNet molecular VMC.
  - DeepSolid solid-state VMC.
- Demonstrates how to wrap different VMC backends behind a common API.
- Supports observables such as energy, density, force, stress, and polarization.

### Important Limitation

NetObs is an observable/evaluation framework, not a training backend for
SolidNES Phase 1.

### Implication For SolidNES

NetObs is valuable mainly as an architectural reference for an adapter layer.
Its `Adaptor` pattern is close to what SolidNES eventually needs:

```text
restore checkpoint
call wavefunction
call local kinetic energy
call local potential energy
make walking step
```

## Comparison Table

| Criterion | FermiNet | DeepSolid | NetObs |
| --- | --- | --- | --- |
| Real solid ground state | Partial PBC support, no real-solid example found | Strongest | Evaluation only |
| Diamond path | Requires custom PBC config | Diamond config available | Depends on backend |
| NES-VMC | Strong, molecular | Not present | Not training |
| PBC + excited states | Explicitly not implemented | Not present | Not training |
| PBC pseudopotential | Explicitly not implemented | Unclear/insufficient in public path | Depends on backend |
| Active maintenance | High | Lower | Moderate |
| First smoke-test fit | Medium | High | Low |
| Long-term NES-VMC fit | High | Medium after large modification | Adapter reference |

## First Backend Decision

For `configs/experiment/diamond_c_deepsolid_ground_smoke.yaml`, target
DeepSolid first.

However, the target must carry an explicit caveat:

```text
DeepSolid-first is for the periodic ground-state workflow, not yet for
the eventual excited-state/NES-VMC extension.
```

The first smoke path should answer:

- Can we run a small periodic ground-state job or at least a config/import
  smoke test?
- Can we parse DeepSolid output into SolidNES results folders?
- What electron count is actually being used?
- What backend objects are available for adapter-level reuse?

## Required Follow-Up

Before claiming an excited-state MVP is ready, resolve:

- Whether the ground-state backend path is reproducible and well-instrumented.
- How to port FermiNet NES-VMC state machinery into a periodic backend.

## Practical Next Step

Create a SolidNES backend note and config bridge for DeepSolid:

```text
scripts/backends/deepsolid_run_config.md
configs/model/deepsolid_detnet_small.yaml
configs/experiment/diamond_c_deepsolid_ground_smoke.yaml
```

No production run should be launched until the environment and dependency plan
for DeepSolid is confirmed.
