# Carbon Diamond Fixed-Checkpoint Evaluation

Date: 2026-05-21 22:51 CST

## Goal

Run a more refined post-training calculation for the current carbon diamond
primitive-cell `sto-3g` model by evaluating the final trained checkpoint without
further neural-network parameter updates.

This separates two effects:

- Training/optimizer noise from changing parameters.
- Monte Carlo sampling uncertainty for one fixed wavefunction.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_evaluate_ckpt4999_batch96_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_evaluate_ckpt4999_batch96_mcmc20.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_evaluation_mcmc20.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/sto3g/training/0012_deepsolid_validation_iter5000_batch96_mcmc12_full/results/checkpoints`
- Target checkpoint:
  `qmcjax_ckpt_004999.npz`
- Output directory:
  `tasks/phase1_diamond_c/sto3g/evaluation/0013_deepsolid_evaluate_ckpt4999_batch96_mcmc20/results`

Key settings:

- `optimizer: none`
- `batch_size: 96`
- `steps_per_iteration: 20`
- `iterations: 10000`
- `pretrain_iterations: 0`
- `burn_in: 0`

DeepSolid restores checkpoint step `4999`, internally starts at step `5000`,
and evaluates through step `9999`, giving 5000 fixed-parameter samples.

## Adapter Change

Added optional `output.restore_checkpoint_dir` support in
`src/solidnes/backends/deepsolid_adapter.py`, mapped to
`cfg.log.restore_path`. The adapter now records `restore_path` in its summary
and fails early if the configured restore directory does not exist.

## SLURM Submission

Submitted as job `120750`.

Planner selected:

- Node: `gpu004`
- Partition: `amdgpu40g`
- GPU: `4 x A100 40GB`
- CPU: `64`
- Walltime: `01:00:00`

Runtime log confirmed:

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `SOLIDNES_EFFECTIVE_CPUS=64`
- `SOLIDNES_EFFECTIVE_GPUS=4`
- `SOLIDNES_CPUS_PER_GPU_LIST=16:16:16:16`
- `SOLIDNES_CPU_BATCHES=gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63`
- `OMP_NUM_THREADS=64`
- `jax_default_backend=gpu`
- `jax_devices=[cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]`

## Completed Result

Job `120750` completed successfully:

- State: `COMPLETED 0:0`
- Elapsed: `00:06:09`
- Rows: `5000`
- Evaluation steps: `5000` through `9999`

Generated outputs:

- `tasks/phase1_diamond_c/sto3g/evaluation/0013_deepsolid_evaluate_ckpt4999_batch96_mcmc20/results/validation/evaluation_detailed_summary.json`
- `tasks/phase1_diamond_c/sto3g/evaluation/0013_deepsolid_evaluate_ckpt4999_batch96_mcmc20/results/validation/evaluation_detailed_summary.md`
- `tasks/phase1_diamond_c/sto3g/evaluation/0013_deepsolid_evaluate_ckpt4999_batch96_mcmc20/results/validation/evaluation_evolution.png`

Main statistics:

- All-sample mean energy: `-72.037661360658 Ha`
- Gap to HF: `+1.966535370943 Ha`
- All-sample naive stderr: `0.011909 Ha`
- All-sample block stderr: `0.021710 Ha`
- Second-half mean energy: `-71.995671542752 Ha`
- Second-half gap to HF: `+2.008525188849 Ha`
- Last-1000 mean energy: `-72.012440707173 Ha`
- Last-1000 gap to HF: `+1.991756024428 Ha`
- Mean variance: `77.347458425685`
- Mean `pmove`: `0.516438437500`
- Minimum single-sample energy: `-77.745445268553 Ha` at step `7274`
- Maximum single-sample energy: `-68.969252305581 Ha` at step `7134`

Interpretation:

The fixed-checkpoint estimate is stable near `-72.0 Ha`, and the acceptance
rate is healthy. The single minimum is not the estimator; it is a noisy local
energy sample. The meaningful block-mean estimate remains about `2 Ha` above
the same-cell HF reference, so this validates the workflow and improves the
estimate, but it is still not an accuracy pass.

## Follow-Up

The next useful accuracy lever is not more fixed-parameter sampling. It should
be either a continued optimization from checkpoint `004999` using a lower
learning rate, or a deliberate physics upgrade such as basis/supercell after the
primitive-cell workflow is accepted.
