# Reference Notes: VMC_reproduce

Repository:

```text
https://github.com/voynova288/VMC_reproduce
```

## High-Level Assessment

VMC_reproduce is a reproducible VMC research repository. It is not simply a
thin script around a fully external model. It implements project-specific model
and physics code, while using mature libraries for autodiff, neural-network
layers, sampling infrastructure, optimizers, and checkpointing.

## What It Builds Itself

- Periodic input features.
- SlaterNet-style baseline.
- Psiformer-style attention model.
- Determinant-sum wavefunction.
- Continuum Hamiltonian pieces.
- Local-energy logic.
- VMC workflows.
- Metrics and checkpoint handling.
- Slurm entry points.

## What It Uses From Libraries

- JAX for autodiff, vectorization, JIT, and device parallelism.
- Flax for neural-network modules and layers.
- NetKet for continuous Hilbert space, MCState, sampler, and operator
  integration in the `vmc/` path.
- Optax for first-order optimizers.
- KFAC-related packages or local wrappers for second-order optimization.
- YAML for config loading.

## Useful Patterns For SolidNES

Borrow these ideas:

- Split configs by role: experiment, system, model, sampler, train.
- Keep scripts thin; put orchestration in workflow modules.
- Use README files as handoff documents for each experiment.
- Track implemented surface, current limits, local checks, Slurm usage, and
  metrics extraction.
- Exclude generated results and raw logs from version control.

## Patterns To Avoid For SolidNES

Do not copy these patterns directly:

- Two sibling code trees such as `pre_vmc/` and `vmc/` before SolidNES needs
  them.
- Reimplementing the full ab initio local-energy engine before backend survey.
- Letting production configs multiply before the MVP is stable.

## SolidNES Translation

For SolidNES, the analogous pattern should be:

```text
SolidNES owns:
  configs
  experiments
  records
  workflows
  diagnostics
  analysis

External or adapted backends provide:
  ansatz
  sampling
  local energy
  optimizer
  checkpoint internals
```

