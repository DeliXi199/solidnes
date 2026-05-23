# Carbon Diamond LR2e-4 Continuation

Date: 2026-05-21

## Goal

Test whether the current medium DeepSolid wavefunction is still
optimization-limited before moving to a larger network.

The previous fixed-checkpoint evaluation gave a stable estimate near
`-72.04 Ha`, about `+1.97 Ha` above the same-cell HF reference. More
fixed-parameter sampling is therefore not the next useful lever. This run
continues parameter optimization from the final 5000-step checkpoint with a
gentler learning rate.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_continue_mcmc20.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/sto3g/training/0012_deepsolid_validation_iter5000_batch96_mcmc12_full/results/checkpoints`
- Target checkpoint:
  `qmcjax_ckpt_004999.npz`
- Output directory:
  `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results`

Key settings:

- `optimizer: adam`
- `learning_rate: 0.0002`
- `learning_rate_delay: 5000.0`
- `learning_rate_decay: 1.0`
- `batch_size: 96`
- `steps_per_iteration: 20`
- `iterations: 10000`
- `pretrain_iterations: 0`
- `clip_local_energy: 5.0`

DeepSolid restores checkpoint step `4999`, internally starts at step `5000`,
and trains through step `9999`, giving 5000 continuation optimization steps.

## Submission

First submission to `intelgpu80g` failed because that partition currently has
only `gpu:2` available for the relevant node configuration. The task was
resubmitted to A100 partitions:

- SLURM job: `120759`
- Partitions: `amdgpu40g,amdgpu80g`
- Requested resources: `gpu:4`, `32 CPU`, `01:00:00`
- Initial state: `PENDING (Resources)`

The request exports CPU batches for four GPUs:

- `SOLIDNES_CPUS_PER_GPU_LIST=8:8:8:8`
- `SOLIDNES_CPU_BATCHES=gpu0:0-7;gpu1:8-15;gpu2:16-23;gpu3:24-31`

Per user request to avoid queueing, pending job `120759` was canceled and the
run was retargeted to the currently available high-resource node:

- SLURM job: `120760`
- Partition: `gpu4090_128`
- Node: `gpu40904`
- Requested resources: `gpu:6`, `96 CPU`, `04:00:00`
- Status after submission: `RUNNING`
- Visible GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- CPU batches:
  `gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63;gpu4:64-79;gpu5:80-95`

To support this, the checkpoint restore shim now reshards restored walkers and
replicated parameter pytrees when the local GPU count differs from the GPU count
stored in the checkpoint. This lets the 4-GPU checkpoint run on 6 GPUs while
keeping `batch_size=96`.

## Success Criteria

- If the continuation mean moves well below `-72.0 Ha`, the current medium
  model was not fully optimized.
- If it remains near `-72.0 Ha`, the likely next lever is model capacity:
  larger hidden layers and more determinants.

## Completed Result

Job `120760` completed successfully:

- State: `COMPLETED 0:0`
- Elapsed: `00:08:32`
- Node: `gpu40904`
- Resources: `6 x RTX 4090`, `96 CPU`
- Rows: `5000`
- Steps: `5000` through `9999`
- Final checkpoint: `qmcjax_ckpt_009999.npz`

Generated outputs:

- `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/validation/training_summary.json`
- `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/validation/training_summary.md`
- `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/validation/continuation_detailed_summary.json`
- `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/validation/continuation_detailed_summary.md`
- `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/validation/continuation_evolution.png`

Main statistics:

- First energy: `-72.643448069218 Ha`
- Last energy: `-73.976436599639 Ha`
- Last-step gap to HF: `+0.027760131962 Ha`
- All-run mean: `-72.788484001366 Ha`
- Second-half mean: `-73.095331312002 Ha`
- Last-1000 mean: `-73.251976787411 Ha`
- Last-500 mean: `-73.311110395212 Ha`
- Last-100 mean: `-73.425988733963 Ha`
- Second-half gap to HF: `+0.908865419599 Ha`
- Last-1000 gap to HF: `+0.752219944190 Ha`
- Last-500 gap to HF: `+0.693086336389 Ha`
- Mean `pmove`: `0.527377916667`
- Mean variance: `57.286423717984`

Interpretation:

The low-learning-rate continuation worked. The medium model was not simply
capacity-limited at `-72 Ha`; it still had significant optimization headroom.
The final training point is very close to HF for the same setup, but a single
step remains noisy. The next required check is a fixed-parameter evaluation of
`qmcjax_ckpt_009999.npz`.
