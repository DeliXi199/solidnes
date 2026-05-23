# Progress: Carbon Diamond Pretrain Validation

Date: 2026-05-21

## Summary

SolidNES tested whether DeepSolid HF-target pretraining reduces the large gap
between the short carbon-diamond VMC validation runs and the same-cell HF
baseline.

The answer is: slightly, but not enough for an accuracy claim.

## Added Configs

```text
configs/train/ground_state_deepsolid_validation_pretrain.yaml
configs/experiment/diamond_c_deepsolid_validation_pretrain.yaml
```

Adapter support was added for:

```text
pretrain_method
pretrain_learning_rate
pretrain_steps
```

The validation summary script now reports tail averages and block standard
errors.

## Run

SLURM job:

```text
120722
state COMPLETED
exit code 0:0
elapsed 00:01:40
partition intelgpu80g
node gpu001
JAX devices [cuda(id=0), cuda(id=1)]
```

Configuration:

```text
batch_size: 16
pretrain_method: net
pretrain_iterations: 100
pretrain_learning_rate: 0.0003
iterations: 100
learning_rate: 0.003
```

## Result

HF reference:

```text
E_HF = -74.0041967316 Ha
```

Training summary:

```text
rows: 100
energy_first: -32.7957561564 Ha
energy_last: -41.9058313886 Ha
energy_min: -49.8937193981 Ha
energy_delta: -9.11007523221 Ha
variance_delta: -217.896529572
pmove_mean: 0.5315
pmove_range: [0.3625, 0.6875]
tail_energy_mean: -38.1650674126 Ha
tail_energy_stderr: 0.475616043807 Ha
block_energy_stderr: 0.919585322207 Ha
last_minus_hf: +32.098365343 Ha
min_minus_hf: +24.1104773335 Ha
tail_mean_minus_hf: +35.839129319 Ha
```

Report:

```text
tasks/phase1_diamond_c/sto3g/training/0008_deepsolid_validation_pretrain/results/validation/training_summary.md
```

## Interpretation

Pretraining improved the best observed energy relative to the previous stable
run:

```text
stable min_minus_hf:   +24.9308470482 Ha
pretrain min_minus_hf: +24.1104773335 Ha
```

This is a real but small improvement. The result is still far from the HF
baseline and should not be described as accurate or converged.

## Next

The next accuracy step should increase representation and optimization quality:

```text
1. Add a medium DeepSolid detnet config.
2. Run 500-1000 iterations with checkpoint continuation.
3. Compare tail/block mean, not just minimum energy.
4. Only after the sto-3g harness behaves, try ccpvdz.
```
