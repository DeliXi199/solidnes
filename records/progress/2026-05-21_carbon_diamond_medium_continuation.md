# Progress: Carbon Diamond Medium Continuation

Date: 2026-05-21

## Summary

SolidNES continued the carbon diamond `sto-3g` medium DeepSolid validation from
the 500-step checkpoint to 1000 total Adam steps.

The continuation improved the tail mean and reduced the HF gap, but the result
remains far from the same-cell HF reference. This is still a validation and
stabilization run, not a physical accuracy claim.

## Run

Continuation config:

```text
configs/experiment/diamond_c_deepsolid_validation_medium_continue_1000.yaml
```

The continuation reuses the medium output directory:

```text
tasks/phase1_diamond_c/sto3g/training/0009_deepsolid_validation_medium/results/
```

SLURM job:

```text
120739
state COMPLETED
exit code 0:0
elapsed 00:01:16
node gpu001
JAX devices [cuda(id=0), cuda(id=1)]
```

Checkpoint created:

```text
tasks/phase1_diamond_c/sto3g/training/0009_deepsolid_validation_medium/results/checkpoints/qmcjax_ckpt_000999.npz
```

## Summary Metrics

The training summary was regenerated from the base medium experiment config so
it covers all rows in the shared medium `train_stats.csv`.

```text
rows: 1000
energy_first: -31.3373461704 Ha
energy_last: -54.2864781349 Ha
energy_min: -81.2861868448 Ha
energy_delta: -22.9491319645 Ha
variance_delta: -18.1178100736
pmove_mean: 0.524675
tail_energy_mean: -51.334478675 Ha
tail_energy_stderr: 0.221863026526 Ha
block_energy_stderr: 0.542992150354 Ha
tail_mean_minus_hf: +22.6697180566 Ha
```

Report:

```text
tasks/phase1_diamond_c/sto3g/training/0009_deepsolid_validation_medium/results/validation/training_summary.md
```

## Interpretation

The medium continuation improved the tail mean relative to the 500-step medium
run:

```text
500-step medium tail mean:  -47.1335720024 Ha
1000-step medium tail mean: -51.3344786750 Ha
```

The same-cell HF baseline remains much lower:

```text
E_HF = -74.0041967316 Ha
1000-step tail mean - HF = +22.6697180566 Ha
```

The very low single-step minimum remains unsuitable as a pass criterion because
the tail and block statistics still show a large average gap. Future accuracy
checks should continue to use tail/block estimates, not minima.

## Next

Continue stabilization before changing the physics target:

```text
1. Continue from checkpoint 000999 with a lower learning rate.
2. Add checkpoint walker resize or resampling before increasing batch size.
3. Only after tail/block metrics stabilize, consider larger basis or excited
   state scaffolding.
```
