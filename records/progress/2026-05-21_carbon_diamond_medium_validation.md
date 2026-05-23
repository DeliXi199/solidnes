# Progress: Carbon Diamond Medium Validation

Date: 2026-05-21

## Summary

SolidNES added checkpoint continuation support and ran a medium DeepSolid detnet
validation for carbon diamond `sto-3g`.

The continuation and medium runs both completed, and the medium tail mean is
better than the small model. The result is still not accurate or converged.

## Checkpoint Continuation

The DeepSolid compatibility shim now patches restore as well as save:

```text
src/solidnes/backends/deepsolid_compat.py
```

The restore shim unwraps object-array checkpoints and restores walkers,
parameters, and MCMC width. Adam optimizer state is reset intentionally because
the legacy DeepSolid restore path reconstructs modern Optax `MultiStepsState`
incorrectly.

Continuation config:

```text
configs/experiment/diamond_c_deepsolid_validation_pretrain_continue_200.yaml
```

SLURM job:

```text
120729
state COMPLETED
exit code 0:0
elapsed 00:00:50
node gpu001
```

Result after appending to the small pretrain run:

```text
rows: 200
energy_last: -44.230042186 Ha
tail_energy_mean: -42.9822478613 Ha
block_energy_stderr: 0.510318151229 Ha
tail_mean_minus_hf: +31.0219488703 Ha
```

Checkpoint created:

```text
tasks/phase1_diamond_c/sto3g/training/0008_deepsolid_validation_pretrain/results/checkpoints/qmcjax_ckpt_000199.npz
```

## Medium Model

Added model:

```text
configs/model/deepsolid_detnet_medium.yaml
```

Medium validation config:

```text
configs/experiment/diamond_c_deepsolid_validation_medium.yaml
```

Settings:

```text
hidden_dims: ((128, 32), (128, 32), (128, 32))
determinants: 8
batch_size: 16
pretrain_iterations: 200
iterations: 500
learning_rate: 0.002
```

SLURM job:

```text
120735
state COMPLETED
exit code 0:0
elapsed 00:01:47
node gpu001
JAX devices [cuda(id=0), cuda(id=1)]
```

Summary:

```text
rows: 500
energy_first: -31.3373461704 Ha
energy_last: -48.1544661922 Ha
energy_min: -81.2861868448 Ha
energy_delta: -16.8171200218 Ha
variance_delta: +204.862834495
pmove_mean: 0.52415
tail_energy_mean: -47.1335720024 Ha
tail_energy_stderr: 0.295293670859 Ha
block_energy_stderr: 1.03792112137 Ha
tail_mean_minus_hf: +26.8706247292 Ha
```

Report:

```text
tasks/phase1_diamond_c/sto3g/training/0009_deepsolid_validation_medium/results/validation/training_summary.md
```

## Interpretation

The medium model improves the tail mean relative to the small-model
continuation:

```text
small continuation tail mean: -42.9822478613 Ha
medium tail mean:             -47.1335720024 Ha
```

However, the medium tail mean remains far above the HF baseline:

```text
E_HF = -74.0041967316 Ha
medium tail mean - HF = +26.8706247292 Ha
```

The very low single minimum is not a pass criterion because the same run has
large variance spikes. Tail and block statistics remain the relevant accuracy
diagnostics.

## Next

Do not move to `ccpvdz` or physical accuracy claims yet. First stabilize this
Hamiltonian:

```text
1. Lower Adam learning rate further.
2. Increase batch size.
3. Continue from the medium checkpoint for longer.
4. Compare only tail/block mean and uncertainty.
```
