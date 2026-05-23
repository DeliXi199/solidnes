# Carbon Diamond cc-pVDZ Continue To 30000

Date: 2026-05-22

## Goal

Continue the cc-pVDZ paper-net Adam run from checkpoint
`qmcjax_ckpt_009999.npz` to 30000 total VMC steps.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000.yaml`
- Model:
  `configs/model/deepsolid_detnet_paper_c_diamond.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_ccpvdz_continue_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/checkpoints`
- Output directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results`

Key settings:

```text
basis: ccpvdz
restore: qmcjax_ckpt_009999
target iterations: 30000
new trained steps: 10000 through 29999
batch_size: 384
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
optimizer: adam
learning_rate: 0.0001
pretrain_iterations: 0
mcmc_burn_in: 0
mcmc_steps_per_iteration: 20
seed: 20260522
```

The same-experiment HF reference was generated:

```text
converged: True
E_HF = -74.9757591792 Ha
```

## Submission

- SLURM job: `121338`
- Job name: `solidnes-ccpvdz-cont30000`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Requested resources: exclusive node, `8 x RTX 4090`, `128 CPU`, `02:00:00`
- Internal timeout: `7200` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt9999_iter30000_plan.json`
- Log:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/logs/slurm/solidnes-ccpvdz-cont30000_121338.log`
- Err:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/logs/slurm/solidnes-ccpvdz-cont30000_121338.err`

Initial status:

```text
JOBID  ST  TIME  NODELIST
121338 R   0:21  gpu40903
```

## Follow-Up

After completion, summarize with `scripts/validation/summarize_deepsolid_validation.py`,
plot the evolution, and compare last/min/tail energy against
`E_HF = -74.9757591792 Ha`.

## Completed Result

Job `121338` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 01:55:56
start: 2026-05-22T12:23:57
end: 2026-05-22T14:19:53
node: gpu40903
resources: 8 x RTX 4090, 128 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/checkpoints/qmcjax_ckpt_029999.npz
tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/validation/training_summary.json
tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/validation/training_summary.md
tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/validation/training_evolution.png
```

Main statistics:

```text
rows: 20000
first step: 10000
last step: 29999
first energy: -74.3952468354 Ha
last energy: -74.7065472189 Ha
energy delta: -0.3113003835 Ha
first variance: 21.3084702461
last variance: 15.2261081901
mean pmove: 0.5391915951
tail mean, last 10000 rows: -74.9086210088 Ha
tail block stderr, 5 blocks: 0.0211787376 Ha
```

Comparison to same-cell `ccpvdz` HF:

```text
E_HF: -74.9757591792 Ha
last minus HF: +0.2692119603 Ha
tail mean minus HF: +0.0671381703 Ha
```

Interpretation:

This is a major convergence improvement. The last-step value remains noisy and
`+0.269 Ha` above HF, but the last-half tail mean is now only about
`+0.067 Ha` above the same-cell `ccpvdz` HF reference. The five tail block
means are still descending:

```text
-74.8401401972
-74.8841030053
-74.9185487930
-74.9420612955
-74.9582517531 Ha
```

The final block is only about `+0.0175 Ha` above HF. This suggests continued
optimization is still useful, but the next decision should include a fixed
checkpoint evaluation or a shorter continuation with careful statistics.
