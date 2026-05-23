# Carbon Diamond cc-pVDZ Paper-Net Sanity Run

Date: 2026-05-22

## Goal

Start the first neural-network VMC task after validating the C-diamond
`ccpvdz` HF baseline. This is a short Adam sanity run, not yet a paper-level
KFAC benchmark.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity.yaml`
- Model:
  `configs/model/deepsolid_detnet_paper_c_diamond.yaml`
- Sampler:
  `configs/sampler/metropolis_deepsolid_ccpvdz_training_mcmc20.yaml`
- Train:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity.yaml`
- DeepSolid template:
  `DeepSolid/config/diamond.py`
- Template input:
  `C,C,3.57,1,ccpvdz`

Key settings:

```text
basis: ccpvdz
batch_size: 384
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
optimizer: adam
learning_rate: 0.0003
pretrain_iterations: 1000
iterations: 1000
mcmc_burn_in: 1000
mcmc_steps_per_iteration: 20
seed: 20260522
params_seed: 20260744
```

The same-experiment HF reference was generated successfully:

```text
converged: True
E_HF = -74.9757591792 Ha
```

Reference file:

```text
tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/validation/pyscf_pbc_hf_reference.json
```

## Validation Before Submission

DeepSolid config build passed:

```text
nelectron: 12
nelec: (6, 6)
basis: ccpvdz
batch_size: 384
iterations: 1000
pretrain_iterations: 1000
mcmc_burn_in: 1000
mcmc_steps_per_iteration: 20
hidden_dims: ((256, 32), (256, 32), (256, 32), (256, 32))
determinants: 8
```

SLURM dry-run passed with a minimum request of 6 GPUs and 96 CPUs.

## Submitted Run

- SLURM job: `121231`
- Job name: `solidnes-ccpvdz-paper-b384`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Requested resources: exclusive node, `8 x RTX 4090`, `128 CPU`, `03:00:00`
- Internal timeout: `10800` seconds
- Plan:
  `tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/outputs/slurm_plans/deepsolid_ccpvdz_paper_b384_mcmc20_adam_sanity_plan.json`
- Log:
  `tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/logs/slurm/solidnes-ccpvdz-paper-b384_121231.log`
- Err:
  `tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/logs/slurm/solidnes-ccpvdz-paper-b384_121231.err`

Startup checks passed:

```text
jax_default_backend=gpu
jax_devices=[cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3),
             cuda(id=4), cuda(id=5), cuda(id=6), cuda(id=7)]
converged SCF energy = -74.9757591791533
```

At the first status check, the job was running:

```text
JOBID  ST  TIME  NODELIST
121231 R   1:19  gpu40903
```

No `train_stats.csv` had been written yet at that checkpoint, so the run was
still in startup, HF, JIT, burn-in, or HF-target pretraining rather than the
logged VMC-iteration phase.

## Follow-Up

When the job finishes or starts writing training stats:

1. Summarize the run with `scripts/validation/summarize_deepsolid_validation.py`.
2. Compare last/min/tail energies against `E_HF = -74.9757591792 Ha`.
3. Check `pmove`, variance, and whether the first cc-pVDZ Adam trajectory is
   stable enough to scale to a longer run.

## Completed Result

Job `121231` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 00:09:27
start: 2026-05-22T10:40:33
end: 2026-05-22T10:50:00
node: gpu40903
resources: 8 x RTX 4090, 128 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/checkpoints/qmcjax_ckpt_000999.npz
tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/validation/training_summary.json
tasks/phase1_diamond_c/ccpvdz/training/0019_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity/results/validation/training_summary.md
```

Main statistics:

```text
rows: 1000
first step: 0
last step: 999
first energy: -70.8880404654 Ha
last energy: -72.8788682535 Ha
minimum energy: -74.7238269243 Ha
energy delta: -1.9908277881 Ha
first variance: 187.6559951074
last variance: 38.6162605732
mean pmove: 0.5184355469
tail mean, last 500 rows: -72.6488795840 Ha
tail block stderr, 5 blocks: 0.1523379475 Ha
```

Comparison to same-cell `ccpvdz` HF:

```text
E_HF: -74.9757591792 Ha
last minus HF: +2.0968909257 Ha
minimum minus HF: +0.2519322548 Ha
tail mean minus HF: +2.3268795952 Ha
```

Interpretation:

The first `ccpvdz` paper-net Adam task is an execution and stability pass. It
completed all 1000 VMC iterations, produced finite statistics, kept useful MCMC
acceptance, and reduced both energy and variance. It is not an accuracy pass:
the tail mean remains about `2.33 Ha` above the same-cell HF reference. The
single minimum-energy row is much closer to HF, but the block means are still
drifting, so the next run should continue or lengthen training rather than
claim convergence.
