# Carbon Diamond cc-pVDZ Continue To 10000

Date: 2026-05-22

## Goal

Continue the first cc-pVDZ paper-net Adam run from checkpoint
`qmcjax_ckpt_000999.npz` to 10000 total VMC steps.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000.yaml`
- Model:
  `configs/model/deepsolid_detnet_paper_c_diamond.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_ccpvdz_continue_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000.yaml`
- Restore directory:
  `tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/checkpoints`
- Output directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results`

Key settings:

```text
basis: ccpvdz
restore: qmcjax_ckpt_000999
target iterations: 10000
new trained steps: 1000 through 9999
batch_size: 384
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
optimizer: adam
learning_rate: 0.0002
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

- SLURM job: `121253`
- Job name: `solidnes-ccpvdz-cont10000`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Requested resources: exclusive node, `8 x RTX 4090`, `128 CPU`, `02:00:00`
- Internal timeout: `7200` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/outputs/slurm_plans/deepsolid_ccpvdz_paper_continue_ckpt999_iter10000_plan.json`
- Log:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/logs/slurm/solidnes-ccpvdz-cont10000_121253.log`
- Err:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/logs/slurm/solidnes-ccpvdz-cont10000_121253.err`

Initial status:

```text
JOBID  ST  TIME  NODELIST
121253 R   0:18  gpu40903
```

Startup log confirmed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3),
             cuda(id=4), cuda(id=5), cuda(id=6), cuda(id=7)]
```

## Follow-Up

After completion, summarize with `scripts/validation/summarize_deepsolid_validation.py`
and compare last/min/tail energy against `E_HF = -74.9757591792 Ha`.

## Completed Result

Job `121253` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 00:52:16
start: 2026-05-22T11:11:10
end: 2026-05-22T12:03:26
node: gpu40903
resources: 8 x RTX 4090, 128 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/checkpoints/qmcjax_ckpt_009999.npz
tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/validation/training_summary.json
tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/validation/training_summary.md
tasks/phase1_diamond_c/ccpvdz/continuation/0020_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000/results/validation/training_evolution.png
```

Main statistics:

```text
rows: 9000
first step: 1000
last step: 9999
first energy: -72.8129042284 Ha
last energy: -74.6358875912 Ha
energy delta: -1.8229833628 Ha
first variance: 40.1309635430
last variance: 23.2866413615
mean pmove: 0.5382869936
tail mean, last 4500 rows: -74.2934236537 Ha
tail block stderr, 5 blocks: 0.0585045711 Ha
```

Comparison to same-cell `ccpvdz` HF:

```text
E_HF: -74.9757591792 Ha
last minus HF: +0.3398715880 Ha
tail mean minus HF: +0.6823355254 Ha
```

Interpretation:

The continuation is a strong optimization improvement: the last-step HF gap
shrinks from about `+2.10 Ha` after the 1000-step sanity run to about
`+0.34 Ha` at step 9999. The last-half block means keep drifting downward
(`-74.0942`, `-74.2560`, `-74.2988`, `-74.3918`, `-74.4263 Ha`), so the run is
not converged yet, but it is still making clear progress. Single-row local
energy outliers can go below HF and should not be interpreted as variational
energy estimates.
