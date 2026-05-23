# Progress: Carbon Diamond Batch64 MCMC10 Validation

Date: 2026-05-21

## Summary

SolidNES designed and ran a larger carbon diamond `sto-3g` DeepSolid validation
task that was expected to be roughly 20 minutes on 2 A100 GPUs.

The goal is not to change the physics target. The goal is to reduce Monte Carlo
noise and test whether a larger sampling budget gives a smoother energy
trajectory than the existing batch16 medium run.

The task completed much faster than expected: 800 VMC iterations finished in
2m21s on 2 A100 GPUs. This indicates that the current carbon `sto-3g` medium
model is still too small to occupy the requested GPU resources for a 20-minute
run.

## Configuration

Experiment:

```text
configs/experiment/diamond_c_deepsolid_validation_batch64_mcmc10_20min.yaml
```

Training:

```text
configs/train/ground_state_deepsolid_validation_batch64_mcmc10_20min.yaml
```

Sampler:

```text
configs/sampler/metropolis_deepsolid_validation_mcmc10.yaml
```

Key parameters:

```text
model: deepsolid_detnet_medium
batch_size: 64
mcmc_burn_in: 10
mcmc_steps_per_iteration: 10
pretrain_iterations: 300
iterations: 800
learning_rate: 0.0005
clip_local_energy: 5.0
```

Output directory:

```text
tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/results/
```

## Rationale

The previous 1000-step medium run used:

```text
batch_size: 16
mcmc_steps_per_iteration: 5
```

This design increases the per-step sampling budget:

```text
batch_size: 64
mcmc_steps_per_iteration: 10
```

That should reduce per-step energy noise and improve sampler decorrelation,
at the cost of slower iterations.

The run starts from scratch instead of continuing from checkpoint `000999`
because the existing checkpoint stores batch16 walker arrays. Directly changing
to batch64 on restore would require checkpoint walker resize or resampling.

## Validation

DeepSolid config build passed:

```text
batch_size: 64
iterations: 800
pretrain_iterations: 300
mcmc_burn_in: 10
mcmc_steps_per_iteration: 10
hidden_dims: ((128, 32), (128, 32), (128, 32))
determinants: 8
```

SLURM dry-run passed:

```text
partition: intelgpu80g
gres: gpu:2
time: 00:25:00
job name: solidnes-gpu-b64-mcmc10-20m
plan: tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/outputs/slurm_plans/deepsolid_gpu_batch64_mcmc10_20min_plan.json
```

Submitted run:

```text
job: 120741
state at submission check: RUNNING
partition: intelgpu80g
node: gpu001
gres: gpu:2
cpus: 96
log: tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/logs/slurm/solidnes-gpu-b64-mcmc10-20m_120741.log
err: tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/logs/slurm/solidnes-gpu-b64-mcmc10-20m_120741.err
```

Startup log confirmed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0), cuda(id=1)]
```

Final status:

```text
state: COMPLETED
exit code: 0:0
elapsed: 00:02:21
rows: 800
checkpoint: qmcjax_ckpt_000799.npz
```

## Results

Summary report:

```text
tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/results/validation/training_summary.md
```

Comparison plot:

```text
tasks/phase1_diamond_c/sto3g/training/0010_deepsolid_validation_batch64_mcmc10_20min/results/validation/batch16_vs_batch64_evolution.png
```

Key metrics:

```text
energy_first: -41.7534072150 Ha
energy_last: -61.5558509267 Ha
energy_min: -87.8141915505 Ha
energy_delta: -19.8024437117 Ha
tail_energy_mean: -59.0591455355 Ha
tail_energy_stderr: 0.1686613176 Ha
block_energy_stderr: 1.1367507922 Ha
tail_mean_minus_hf: +14.9450511961 Ha
tail_variance_mean: 380.1651988002
tail_pmove_mean: 0.5210976563
pmove_range: [0.4453125, 0.6140625]
```

Comparison to the previous batch16 medium run:

```text
batch16 tail mean:        -51.3344786750 Ha
batch64/MCMC10 tail mean: -59.0591455355 Ha
improvement:              -7.7246668605 Ha

batch16 tail gap to HF:        +22.6697180566 Ha
batch64/MCMC10 tail gap to HF: +14.9450511961 Ha

batch16 tail residual sd vs 50-step mean:        4.639765 Ha
batch64/MCMC10 tail residual sd vs 50-step mean: 2.498143 Ha
```

The larger sampling budget clearly reduced visible step-to-step energy noise
and improved the tail energy. The tail block means are still drifting downward,
so the block stderr is larger than batch16 and should be interpreted as
non-convergence rather than a clean statistical uncertainty.

## Expected Diagnostic

After the run, compare against the batch16 medium run using:

```text
1. tail_energy_mean
2. block_energy_stderr
3. tail variance mean
4. rolling energy plot smoothness
5. pmove stability
```

Success for this task means lower noise and stable finite statistics, not
physical accuracy. That success criterion was met, but the run is still far
above the same-cell HF reference and should not be called accurate.
