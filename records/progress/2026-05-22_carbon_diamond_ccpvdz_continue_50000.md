# Carbon Diamond cc-pVDZ Continue To 50000

Date: 2026-05-22

## Goal

Continue the cc-pVDZ paper-net Adam run from checkpoint
`qmcjax_ckpt_029999.npz` to 50000 total VMC steps, adding another 20000
training steps.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000.yaml`
- Model:
  `configs/model/deepsolid_detnet_paper_c_diamond.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_ccpvdz_continue_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0021_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000/results/checkpoints`
- Output directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results`

Key settings:

```text
basis: ccpvdz
restore: qmcjax_ckpt_029999
target iterations: 50000
new trained steps: 30000 through 49999
batch_size: 384
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
optimizer: adam
learning_rate: 0.00005
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

## Superseded 8-GPU Submission

- SLURM job: `121397`
- Job name: `solidnes-ccpvdz-cont50000`
- Submission rule: current GPU submission policy via
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh`
- Selected partition set: `h200,amdgpu80g,amdgpu40g,h20,h800`
- Requested resources: `8 GPU`, `128 CPU`, `64 GB`, `03:00:00`
- Internal timeout: `10800` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt29999_iter50000_plan.json`
- Expected log:
  `logs/slurm/solidnes-ccpvdz-cont50000_121397.log`
- Expected err:
  `logs/slurm/solidnes-ccpvdz-cont50000_121397.err`

Initial status:

```text
JOBID   ST  TIME  TIME_LIMIT  NODELIST(REASON)
121397  PD  0:00  3:00:00     (Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)
```

Note: pending job `121396` was canceled before start after a manual
gpu4090-specific reroute was rejected; job `121397` was the first submission
under the current GPU rules.

This 8-GPU submission was later canceled before start.

## Superseded 4-GPU Submission

- SLURM job: `121398`
- Job name: `solidnes-ccpvdz-cont50000-g4`
- Submission rule: current GPU submission policy via
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh`
- Selected partition set: `h200,amdgpu80g,amdgpu40g,h20,h800`
- Requested resources: `4 GPU`, `64 CPU`, `64 GB`, `08:00:00`
- Internal timeout: `28800` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt29999_iter50000_g4_plan.json`
- Expected log:
  `logs/slurm/solidnes-ccpvdz-cont50000-g4_121398.log`
- Expected err:
  `logs/slurm/solidnes-ccpvdz-cont50000-g4_121398.err`

Current status:

```text
JOBID   ST  TIME  TIME_LIMIT  NODELIST(REASON)
121398  PD  0:00  8:00:00     (Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)
```

SLURM detailed scheduling snapshot:

```text
scheduled node: gpu005
predicted start: 2026-05-23T01:50:00
predicted end: 2026-05-23T09:50:00
requested TRES: cpu=64,mem=62.50G,node=1,billing=64,gres/gpu=4
```

This 4-GPU submission was canceled before start when the idle `intelgpu80g`
node was identified as a better fit for a 2-GPU run.

## Active 2-GPU Submission

- SLURM job: `121423`
- Job name: `solidnes-ccpvdz-cont50000-g2`
- Submission rule: current GPU submission policy via
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh`
- Reason: `best_idle_gpu_node`
- Partition: `intelgpu80g`
- Node: `gpu001`
- GPU model key: `a100_80gb`
- Requested resources: exclusive node, `2 GPU`, `96 CPU`, `16:00:00`
- Internal timeout: `57600` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt29999_iter50000_g2_plan.json`
- Log:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/logs/slurm/solidnes-ccpvdz-cont50000-g2_121423.log`
- Err:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/logs/slurm/solidnes-ccpvdz-cont50000-g2_121423.err`

Initial active status:

```text
JOBID   PARTITION    ST  TIME  TIME_LIMIT  NODELIST
121423  intelgpu80g  R   0:15  16:00:00    gpu001
```

## Follow-Up

After completion, summarize with `scripts/validation/summarize_deepsolid_validation.py`,
plot the evolution, and compare last/min/tail energy against
`E_HF = -74.9757591792 Ha`.

## Completed Result

Job `121423` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 01:10:36
start: 2026-05-22T14:42:44
end: 2026-05-22T15:53:20
node: gpu001
resources: 2 GPU on intelgpu80g, 96 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results/checkpoints/qmcjax_ckpt_049999.npz
tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results/validation/training_summary.json
tasks/phase1_diamond_c/ccpvdz/continuation/0023_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000/results/validation/training_summary.md
```

Main statistics:

```text
rows: 20000
first step: 30000
last step: 49999
first energy: -74.8392740274 Ha
last energy: -75.2077802431 Ha
energy delta: -0.3685062156 Ha
first variance: 10.2293105497
last variance: 6.7055957245
variance delta: -3.5237148252
mean pmove: 0.5394034115
tail mean, last 10000 rows: -75.0909943902 Ha
tail block stderr, 5 blocks: 0.0088464229 Ha
```

Comparison to same-cell `ccpvdz` HF:

```text
E_HF: -74.9757591792 Ha
last minus HF: -0.2320210639 Ha
tail mean minus HF: -0.1152352110 Ha
```

The five tail block means are:

```text
-75.0624020112
-75.0784455789
-75.1005591272
-75.1064938400
-75.1070713936 Ha
```

Interpretation:

The 30000-to-50000 continuation completed cleanly and improved the robust
tail estimate by about `0.1824 Ha` relative to the 30000-step tail mean
(`-74.9086210088 Ha`). The final two tail blocks are nearly flat around
`-75.107 Ha`, so the Adam run is much closer to a plateau than at 30000 steps.
The result remains above the DeepSolid paper small-cell VMC value
`-75.4009 Ha`, so this is a successful Adam refinement, not a paper-level
benchmark match.
