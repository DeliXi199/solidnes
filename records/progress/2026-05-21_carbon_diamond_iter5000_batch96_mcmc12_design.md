# Progress: Carbon Diamond Iter5000 Batch96 MCMC12 Design

Date: 2026-05-21

## Summary

SolidNES prepared an iteration-dominant carbon diamond `sto-3g` DeepSolid
validation task.

The design follows the current priority: increase training iterations
aggressively while only mildly increasing the sampling parameters.

## Configuration

Experiment:

```text
configs/experiment/diamond_c_deepsolid_validation_iter5000_batch96_mcmc12.yaml
```

Training:

```text
configs/train/ground_state_deepsolid_validation_iter5000_batch96_mcmc12.yaml
```

Sampler:

```text
configs/sampler/metropolis_deepsolid_validation_mcmc12.yaml
```

Key parameters:

```text
model: deepsolid_detnet_medium
batch_size: 96
mcmc_burn_in: 12
mcmc_steps_per_iteration: 12
pretrain_iterations: 500
iterations: 5000
learning_rate: 0.0003
clip_local_energy: 5.0
```

Output directory:

```text
tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/results/
```

## Rationale

The previous batch64/MCMC10 task showed clear improvement but its tail block
means were still drifting downward:

```text
-55.72, -57.76, -59.19, -60.14, -62.49 Ha
```

That is a convergence signal: more optimization steps are needed before making
stronger conclusions. This task therefore increases:

```text
iterations: 800 -> 5000
batch_size: 64 -> 96
mcmc_steps_per_iteration: 10 -> 12
pretrain_iterations: 300 -> 500
```

The learning rate is reduced to `0.0003` because the run is much longer.

## Validation

DeepSolid config build passed:

```text
batch_size: 96
iterations: 5000
pretrain_iterations: 500
mcmc_burn_in: 12
mcmc_steps_per_iteration: 12
hidden_dims: ((128, 32), (128, 32), (128, 32))
determinants: 8
```

SLURM dry-run passed:

```text
partition: intelgpu80g
gres: gpu:2
cpus-per-task: 16
time: 00:45:00
job name: solidnes-gpu-iter5000-b96-m12
plan: tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/outputs/slurm_plans/deepsolid_gpu_iter5000_batch96_mcmc12_plan.json
```

Submitted run:

```text
job: 120743
state at submission check: RUNNING
partition: intelgpu80g
node: gpu001
gres: gpu:2
cpus allocated by planner: 96
log: tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/logs/slurm/solidnes-gpu-iter5000-b96-m12_120743.log
err: tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/logs/slurm/solidnes-gpu-iter5000-b96-m12_120743.err
```

Startup log confirmed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0), cuda(id=1)]
```

One instantaneous `nvidia-smi` snapshot early in the run showed both A100s in
P0 state with allocated memory, but `utilization.gpu=0%` at that exact moment.
This is not enough to characterize full-run utilization because the snapshot was
taken during early startup/JIT/pretrain timing.

Final status:

```text
state: FAILED
exit code: 124:0
elapsed: 00:05:12
reason: internal DeepSolid process smoke timeout at 300 seconds
rows written: 3197
last step: 3196
final clean checkpoint: none
available checkpoint: qmcjax_ckpt_000000.npz
```

The SLURM walltime was `00:45:00`, so this was not a scheduler time-limit
failure. The failure came from the wrapper default:

```text
SOLIDNES_TIMEOUT_SECONDS=300
```

## Partial Results

Even though the run did not complete all 5000 iterations, the partial result is
highly informative.

Summary report:

```text
tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/results/validation/training_summary.md
```

Three-run comparison plot:

```text
tasks/phase1_diamond_c/sto3g/training/0011_deepsolid_validation_iter5000_batch96_mcmc12/results/validation/three_run_comparison.png
```

Key metrics:

```text
rows: 3197
last_step: 3196
energy_first: -57.7949294657 Ha
energy_last: -72.1752396199 Ha
energy_min: -75.8069906299 Ha
energy_delta: -14.3803101541 Ha
tail_energy_mean: -70.8751459188 Ha
tail_energy_stderr: 0.0299976187 Ha
block_energy_stderr: 0.2284807761 Ha
tail_mean_minus_hf: +3.1290508128 Ha
tail_variance_mean: 120.8908283939
tail_pmove_mean: 0.5223805243
pmove_range: [0.4201388889, 0.6032986111]
```

Comparison to earlier runs:

```text
batch16 tail mean:          -51.3344786750 Ha
batch64/MCMC10 tail mean:   -59.0591455355 Ha
iter5000 partial tail mean: -70.8751459188 Ha

batch16 tail gap to HF:          +22.6697180566 Ha
batch64/MCMC10 tail gap to HF:   +14.9450511961 Ha
iter5000 partial tail gap to HF:  +3.1290508128 Ha

batch16 tail residual sd vs 50-step mean:          4.639765 Ha
batch64/MCMC10 tail residual sd vs 50-step mean:   2.498143 Ha
iter5000 partial tail residual sd vs 50-step mean: 1.084754 Ha
```

The result strongly supports the user's intuition that increasing iterations is
currently the dominant improvement lever. The next submission should use the
same physics/config shape but set a longer internal timeout and a much smaller
checkpoint interval.

## Expected Diagnostic

After the run, compare against batch64/MCMC10 using:

```text
1. tail_energy_mean
2. block_energy_means and whether they stop drifting downward
3. block_energy_stderr
4. tail residual sd against the 50-step rolling mean
5. tail_mean_minus_hf
6. pmove range and mean
```

Success means the tail enters a more stable plateau or continues improving
without acceptance collapse.
