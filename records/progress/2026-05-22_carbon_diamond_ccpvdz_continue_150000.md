# Carbon Diamond cc-pVDZ Continue To 150000

Date: 2026-05-22

## Goal

Continue the cc-pVDZ paper-net Adam run from checkpoint
`qmcjax_ckpt_049999.npz` to 150000 total VMC steps, adding another 100000
training steps.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000.yaml`
- Model:
  `configs/model/deepsolid_detnet_paper_c_diamond.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_ccpvdz_continue_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results/checkpoints`
- Output directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results`

Key settings:

```text
basis: ccpvdz
restore: qmcjax_ckpt_049999
target iterations: 150000
new trained steps: 50000 through 149999
batch_size: 384
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
optimizer: adam
learning_rate: 0.00002
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

- SLURM job: `122085`
- Job name: `solidnes-ccpvdz-cont150000-auto`
- Submission rule: current GPU auto-sizing policy via
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh`
- Scheduler reason: `best_idle_gpu_node`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model key: `a100_80gb`
- Requested resources: exclusive node, `2 GPU`, `96 CPU`, `16:00:00`
- Internal timeout: `57600` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt49999_iter150000_auto_plan.json`
- Log:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/logs/slurm/solidnes-ccpvdz-cont150000-auto_122085.log`
- Err:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/logs/slurm/solidnes-ccpvdz-cont150000-auto_122085.err`

Initial status:

```text
JOBID   PARTITION    ST  TIME  TIME_LIMIT  NODELIST
122085  intelgpu80g  R   0:20  16:00:00    gpu001
```

## Follow-Up

After completion, summarize with `scripts/validation/summarize_deepsolid_validation.py`,
plot the evolution, and compare the last-half tail mean against:

```text
same-cell cc-pVDZ HF: -74.9757591792 Ha
DeepSolid paper small-cell VMC: -75.4009 Ha
previous 50000-step tail mean: -75.0909943902 Ha
```

## Completed Result

Job `122085` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 05:50:50
start: 2026-05-22T16:29:58
end: 2026-05-22T22:20:48
node: gpu001
resources: 2 GPU on intelgpu80g, 96 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results/checkpoints/qmcjax_ckpt_149999.npz
tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results/validation/training_summary.json
tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results/validation/training_summary.md
tasks/phase1_diamond_c/ccpvdz/continuation/0026_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000/results/validation/training_evolution.png
```

Main statistics:

```text
rows: 100000
first step: 50000
last step: 149999
first energy: -75.2215320914 Ha
last energy: -75.2704621344 Ha
energy delta: -0.0489300430 Ha
first variance: 6.3147795090
last variance: 10.4466143737
mean pmove: 0.5393589987
tail mean, last 50000 rows: -75.1982415149 Ha
tail block stderr, 5 blocks: 0.0033233224 Ha
```

Comparison:

```text
same-cell cc-pVDZ HF: -74.9757591792 Ha
DeepSolid paper small-cell VMC: -75.4009 Ha
tail mean minus HF: -0.2224823357 Ha
tail mean minus paper VMC: +0.2026584851 Ha
tail mean improvement vs 50000-step tail: -0.1072471247 Ha
```

The five tail block means are:

```text
-75.1865111522
-75.1963293320
-75.1993644290
-75.2045070164
-75.2044956447 Ha
```

Interpretation:

The 50000-to-150000 continuation improved the robust tail estimate from
`-75.0909943902 Ha` to `-75.1982415149 Ha`, an additional `0.107247 Ha`
decrease. The final tail blocks flatten near `-75.2045 Ha`, so Adam is still
improving but appears close to another plateau. The run remains about
`0.2027 Ha` above the DeepSolid paper small-cell VMC reference `-75.4009 Ha`,
so more Adam steps alone are unlikely to close the full gap efficiently.
