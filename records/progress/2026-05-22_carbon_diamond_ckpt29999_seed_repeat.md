# Carbon Diamond Checkpoint 29999 Seed Repeat

Date: 2026-05-22

## Goal

Repeat the fixed-parameter evaluation of `qmcjax_ckpt_029999.npz` with a
different sampler RNG seed. This checks whether the previous near-HF fixed
estimate is reproducible under independent Monte Carlo sampling.

## Seed Support

The SolidNES DeepSolid adapter now maps `training.seed` into the DeepSolid
runtime config as `cfg.debug.seed`, with `cfg.debug.params_seed = seed + 222`
unless explicitly overridden. The vendored DeepSolid process entrypoint now
honors those optional seed fields instead of only using the hard-coded
deterministic seeds.

For this repeat:

- Sampler/walker seed: `20260522`
- Parameter seed: `20260744`

For restored fixed-checkpoint evaluation, the parameter seed is recorded but
does not reinitialize the checkpointed network parameters.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000.yaml`
- Train:
  `configs/train/ground_state_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_evaluation_mcmc30.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/sto3g/continuation/0015_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000/results/checkpoints`
- Target checkpoint:
  `qmcjax_ckpt_029999.npz`
- Output directory:
  `tasks/phase1_diamond_c/sto3g/evaluation/0017_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000/results`

Key settings:

- `optimizer: none`
- `batch_size: 96`
- `steps_per_iteration: 30`
- `iterations: 40000`
- `pretrain_iterations: 0`
- `burn_in: 0`

DeepSolid restores checkpoint step `29999`, internally starts at step `30000`,
and evaluates through step `39999`, giving 10000 fixed-parameter samples.

## Submission

- SLURM job: `120977`
- Partition: `gpu4090_128`
- Node: `gpu40904`
- Requested resources: `gpu:6`, `96 CPU`, `02:00:00`
- Initial state: `RUNNING`
- Visible GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- CPU batches:
  `gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63;gpu4:64-79;gpu5:80-95`

## Follow-Up

When the job finishes, compare this seed-repeat estimate against the previous
fixed evaluation:

- all-sample mean and 10-block stderr,
- first-half/second-half means,
- gap to `E_HF = -74.0041967316 Ha`,
- mean variance and `pmove`,
- whether the two independent fixed evaluations agree within statistical
  uncertainty.

## Completed Result

Job `120977` completed successfully:

- State: `COMPLETED 0:0`
- Elapsed: `00:15:55`
- Node: `gpu40904`
- Resources: `6 x RTX 4090`, `96 CPU`
- Rows: `10000`
- Steps: `30000` through `39999`

Generated outputs:

- `tasks/phase1_diamond_c/sto3g/evaluation/0017_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_seed20260522_detailed_summary.json`
- `tasks/phase1_diamond_c/sto3g/evaluation/0017_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_seed20260522_detailed_summary.md`
- `tasks/phase1_diamond_c/sto3g/evaluation/0017_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000/results/validation/fixed_ckpt29999_seed20260522_evolution.png`

Main statistics:

- All-sample mean: `-74.037947329305 Ha`
- All-sample gap to HF: `-0.033750597704 Ha`
- 10-block stderr: `0.007185342404 Ha`
- First-half mean: `-74.030021256712 Ha`
- Second-half mean: `-74.045873401897 Ha`
- Last-1000 mean: `-74.005314297332 Ha`
- Mean `pmove`: `0.537045902778`
- Mean variance: `25.225071003892`

Comparison with the previous fixed evaluation:

- Previous all-sample mean: `-74.021768218405 Ha`
- New all-sample mean: `-74.037947329305 Ha`
- New minus previous: `-0.016179110900 Ha`
- Combined block stderr: `0.009984717289 Ha`
- Difference in combined sigma: `-1.620`
- Equal-weight mean of the two fixed evaluations:
  `-74.029857773855 Ha`
- Equal-weight mean gap to HF: `-0.025661042254 Ha`
- Equal-weight mean stderr estimate: `0.004992358645 Ha`

Interpretation:

The independent seed-repeat evaluation confirms the checkpoint remains near
and slightly below the same-cell HF reference. The new seed gives a lower mean
than the first fixed evaluation, but the difference is only about `1.62`
combined block standard errors, so it is not a clear contradiction. It mainly
shows residual Monte Carlo sampling/autocorrelation noise at roughly the
`10--20 mHa` level for a 10000-row MCMC30 fixed evaluation.
