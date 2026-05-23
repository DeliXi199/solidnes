# Carbon Diamond Checkpoint 9999 20000-Step Continuation

Date: 2026-05-22

## Goal

Add 20000 optimization iterations after the successful `lr2e-4` continuation
that ended at checkpoint `qmcjax_ckpt_009999.npz`.

This tests whether the medium network continues improving after reaching a
near-HF final training point.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000.yaml`
- Train:
  `configs/train/ground_state_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_continue_mcmc20.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/sto3g/continuation/0014_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20/results/checkpoints`
- Target checkpoint:
  `qmcjax_ckpt_009999.npz`
- Output directory:
  `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results`

Key settings:

- `optimizer: adam`
- `learning_rate: 0.0001`
- `learning_rate_delay: 10000.0`
- `learning_rate_decay: 1.0`
- `batch_size: 96`
- `steps_per_iteration: 20`
- `iterations: 30000`
- `pretrain_iterations: 0`
- `clip_local_energy: 5.0`

DeepSolid restores checkpoint step `9999`, internally starts at step `10000`,
and trains through step `29999`, giving 20000 additional optimization steps.

## Submission

- SLURM job: `120805`
- Partition: `gpu4090_128`
- Node: `gpu40904`
- Requested resources: `gpu:6`, `96 CPU`, `04:00:00`
- Initial state: `RUNNING`
- Visible GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- CPU batches:
  `gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63;gpu4:64-79;gpu5:80-95`
- Startup GPU utilization check: about `91--94%` across all six visible GPUs.

## Follow-Up

When the job finishes, summarize:

- tail/block energy means for the 20000-step continuation,
- gap to `E_HF = -74.0041967316 Ha`,
- variance and `pmove`,
- whether the longer low-LR optimization kept improving or plateaued.

## Completed Result

Job `120805` completed successfully:

- State: `COMPLETED 0:0`
- Elapsed: `00:27:36`
- Node: `gpu40904`
- Resources: `6 x RTX 4090`, `96 CPU`
- Rows: `20000`
- Steps: `10000` through `29999`
- Final checkpoint: `qmcjax_ckpt_029999.npz`

Generated outputs:

- `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/validation/training_summary.json`
- `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/validation/training_summary.md`
- `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/validation/long_continuation_detailed_summary.json`
- `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/validation/long_continuation_detailed_summary.md`
- `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/validation/long_continuation_evolution.png`

Main statistics:

- First energy: `-72.766098447140 Ha`
- Last energy: `-73.663077791587 Ha`
- All-run mean: `-73.792729936901 Ha`
- First-half mean: `-73.627377030407 Ha`
- Second-half mean: `-73.958082843396 Ha`
- Last-5000 mean: `-74.014606121362 Ha`
- Last-2000 mean: `-74.028304211607 Ha`
- Last-1000 mean: `-74.032080094067 Ha`
- Last-500 mean: `-74.020123304181 Ha`
- Mean `pmove`: `0.531274713542`
- Mean variance: `33.624980122959`

Relative to `E_HF = -74.0041967316 Ha`:

- All-run gap: `+0.211466794700 Ha`
- Second-half gap: `+0.046113888205 Ha`
- Last-5000 gap: `-0.010409389761 Ha`
- Last-2000 gap: `-0.024107480006 Ha`
- Last-1000 gap: `-0.027883362466 Ha`
- Last-500 gap: `-0.015926572580 Ha`

Interpretation:

The additional 20000 low-LR optimization steps worked very well. The run moved
from a near-HF final training point to a tail trajectory whose block means sit
around the HF line. Because training estimates are noisy and some tail means are
slightly below HF, this should not be declared final accuracy yet. The correct
next step is an independent fixed-checkpoint evaluation of
`qmcjax_ckpt_029999.npz`.
