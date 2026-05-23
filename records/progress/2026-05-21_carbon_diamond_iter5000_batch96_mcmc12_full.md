# Progress: Carbon Diamond Iter5000 Batch96 MCMC12 Full

Date: 2026-05-21

## Summary

SolidNES completed the full 5000-step carbon diamond `sto-3g` DeepSolid
validation.

The run confirms that increasing iteration count was the most important lever
for the current setup. The tail mean moved much closer to the same-cell HF
reference, and the last-step energy is within about 1.6 Ha of HF.

## Run

Experiment:

```text
configs/experiment/diamond_c_deepsolid_validation_iter5000_batch96_mcmc12_full.yaml
```

SLURM job:

```text
120746
state COMPLETED
exit code 0:0
elapsed 00:06:54
node gpu004
allocated TRES cpu=64, gres/gpu=4
```

Outputs:

```text
tasks/phase1_diamond_c/sto3g/training/0012_deepsolid_validation_iter5000_batch96_mcmc12_full/results/
```

Checkpoints:

```text
qmcjax_ckpt_000000.npz
qmcjax_ckpt_001803.npz
qmcjax_ckpt_003880.npz
qmcjax_ckpt_004999.npz
```

## Metrics

```text
rows: 5000
last_step: 4999
energy_first: -55.1440831561 Ha
energy_last: -72.4107097191 Ha
energy_min: -79.3804751803 Ha
energy_delta: -17.2666265630 Ha
tail50_energy_mean: -71.2518826225 Ha
tail20_energy_mean: -71.6459658911 Ha
tail10_energy_mean: -71.7684844048 Ha
hf_energy: -74.0041967316 Ha
tail50_minus_hf: +2.7523141091 Ha
tail20_minus_hf: +2.3582308405 Ha
tail10_minus_hf: +2.2357123268 Ha
last_minus_hf: +1.5934870124 Ha
tail_variance_mean: 113.4888146285
tail_pmove_mean: 0.5212295139
pmove_range: [0.4131944444, 0.6631944444]
```

Tail50 block means:

```text
-70.611076, -71.086662, -71.269743, -71.523447, -71.768484 Ha
```

## Interpretation

The full run is substantially better than the timeout-truncated partial run:

```text
partial tail50: -70.8751459188 Ha
full tail50:    -71.2518826225 Ha
```

The tail block means still drift downward, so the run is improving but not
fully plateaued. The single minimum below HF is not a pass criterion because it
is a noisy single-step outlier.

There is a large variance/energy spike at step 1732:

```text
energy: -45.7010853302 Ha
variance: 63060.1775579747
```

After that spike, the run recovers and the late trajectory is much steadier.
The last 1000 steps have residual standard deviation about `0.923 Ha` against a
50-step rolling mean.

## Artifacts

Summary:

```text
tasks/phase1_diamond_c/sto3g/training/0012_deepsolid_validation_iter5000_batch96_mcmc12_full/results/validation/training_summary.md
```

Single-run plot:

```text
tasks/phase1_diamond_c/sto3g/training/0012_deepsolid_validation_iter5000_batch96_mcmc12_full/results/validation/training_evolution_full.png
```

## Next

The next scientifically useful run should continue this direction: either run
longer at the same learning rate to see whether the tail reaches a plateau, or
continue from checkpoint `004999` with a lower learning rate after adding
batch96 checkpoint restore compatibility.
