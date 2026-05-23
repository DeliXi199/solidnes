# Carbon Diamond Checkpoint 29999 Fixed Evaluation

Date: 2026-05-22

## Goal

Run an independent fixed-parameter evaluation of
`qmcjax_ckpt_029999.npz` after the 20000-step low-learning-rate continuation.

The training tail reached the HF line and even dipped slightly below it in
short tail windows, so this evaluation is needed to separate real fixed
wavefunction quality from training-trajectory Monte Carlo noise.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000.yaml`
- Train:
  `configs/train/ground_state_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_evaluation_mcmc30.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/checkpoints`
- Target checkpoint:
  `qmcjax_ckpt_029999.npz`
- Output directory:
  `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results`

Key settings:

- `optimizer: none`
- `batch_size: 96`
- `steps_per_iteration: 30`
- `iterations: 40000`
- `pretrain_iterations: 0`
- `burn_in: 0`
- `clip_local_energy: 5.0`

DeepSolid restores checkpoint step `29999`, internally starts at step `30000`,
and evaluates through step `39999`, giving 10000 fixed-parameter samples.

## Submission

- SLURM job: `120884`
- Partition: `gpu4090_128`
- Node: `gpu40904`
- Requested resources: `gpu:6`, `96 CPU`, `04:00:00`
- Initial state: `RUNNING`
- Visible GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- CPU batches:
  `gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63;gpu4:64-79;gpu5:80-95`
- Startup GPU utilization check: about `83--85%` across all six visible GPUs.

## Follow-Up

When the job finishes, summarize:

- all/second-half/last-window fixed-parameter energy means,
- block standard errors,
- gap to `E_HF = -74.0041967316 Ha`,
- variance and `pmove`,
- whether the trained checkpoint is genuinely near HF under independent
  evaluation.

## Completed Result

Job `120884` completed successfully:

- State: `COMPLETED 0:0`
- Elapsed: `00:15:48`
- Node: `gpu40904`
- Resources: `6 x RTX 4090`, `96 CPU`
- Rows: `10000`
- Steps: `30000` through `39999`

Generated outputs:

- `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results/validation/evaluation_summary_all.json`
- `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results/validation/evaluation_summary_all.md`
- `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_detailed_summary.json`
- `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_detailed_summary.md`
- `tasks/phase1_diamond_c/sto3g/evaluation/0016_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_evolution.png`

Main statistics:

- All-sample mean: `-74.021768218405 Ha`
- All-sample gap to HF: `-0.017571486804 Ha`
- 10-block stderr: `0.006932923906 Ha`
- First-half mean: `-74.017387225155 Ha`
- First-half gap to HF: `-0.013190493555 Ha`
- Second-half mean: `-74.026149211655 Ha`
- Second-half gap to HF: `-0.021952480054 Ha`
- Mean `pmove`: `0.537692395833`
- Mean variance: `24.389738793891`
- Minimum single-sample energy: `-77.549311987539 Ha` at step `37256`
- Maximum single-sample energy: `-70.097505624739 Ha` at step `35449`

Interpretation:

The independent fixed-checkpoint estimate confirms the trained checkpoint is
near the same-cell HF reference. The all-sample mean is slightly below HF by
about `0.0176 Ha`, with a 10-block standard error of about `0.0069 Ha`.

This should be interpreted as a strong validation result for the current
optimization/sampling workflow on carbon diamond primitive-cell `sto-3g`, not
as a production physical benchmark. The next decision is either to repeat the
fixed evaluation with a different seed/longer sampling, or to move to a physics
upgrade such as larger basis and/or supercell.
