# Carbon Diamond cc-pVDZ Paper-Like KFAC Probe

Date: 2026-05-22

## Objective

Run a short C-diamond cc-pVDZ probe with paper-like DeepSolid parameters while
keeping the VMC training length at 1000 iterations:

- basis: `ccpvdz`
- ansatz: DeepSolid paper C-diamond DetNet, `(256, 32) x 4`
- determinants: `8`
- optimizer: `kfac`
- learning rate: `3e-2`
- batch size: `4096`
- pretrain: `1000`
- burn-in: `1000`
- MCMC steps per iteration: `20`
- VMC iterations: `1000`

## Files

- experiment config:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000.yaml`
- train config:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000.yaml`
- output directory:
  `tasks/phase1_diamond_c/ccpvdz/training/0028_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000/results`
- SLURM plan:
  `tasks/phase1_diamond_c/ccpvdz/training/0028_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000/outputs/slurm_plans/deepsolid_ccpvdz_paper_kfac_b4096_iter1000_plan.json`

## Runtime Fixes

- `scripts/backends/run_deepsolid_process_smoke.py` now neutralizes KFAC tags
  only for non-KFAC runs.
- `src/solidnes/backends/deepsolid_adapter.py` now forwards optional `training.kfac`
  YAML fields into `cfg.optim.kfac`.
- `src/solidnes/slurm_scheduling.py`, `scripts/slurm/plan_slurm_job.py`, and
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh` now support
  `SOLIDNES_GPU_ALLOWED_COUNTS`; this prevents invalid requests such as 3 GPUs
  for `batch_size = 4096`.
- `src/solidnes/backends/deepsolid_compat.py` now patches DeepSolid pretraining
  so pretrain temporarily uses neutral tags, then restores real KFAC tags before
  KFAC optimization.

## Submission History

- Job `125516`: failed immediately because the scheduler selected 3 GPUs and
  DeepSolid requires `batch_size % local_device_count == 0`.
- Job `125517`: selected `intelgpu80g/gpu001` with `2 GPU + 96 CPU`, then failed
  in pretraining because legacy KFAC tags are incompatible with the pmapped
  pretrain path on JAX `0.4.30`.
- Job `125574`: submitted after both fixes. It is running on
  `intelgpu80g/gpu001` with `2 GPU + 96 CPU`, exclusive node allocation, and
  `12:00:00` wall time.

## Final Status

Job `125574` ended `FAILED` with exit code `1:0` after `00:07:10`.

Logs confirm:

- `jax_default_backend=gpu`
- `jax_devices=[cuda(id=0), cuda(id=1)]`
- same-cell cc-pVDZ HF converged at `-74.9757591791533 Ha`
- the earlier pretrain tag error was avoided
- checkpoint `qmcjax_ckpt_000000.npz` was written after pretraining

The run failed when DeepSolid initialized KFAC:

```text
File ".../DeepSolid/process.py", line 235, in process
  opt_state = optimizer.init(params, subkeys, data)
...
File ".../DeepSolid/network.py", line 443, in linear_layer
  return curvature_tags_and_blocks.register_repeated_dense(y, x, w, b)
TypeError: cannot unpack non-iterable ShapedArray object
```

No `train_stats` file was produced, so no KFAC VMC iterations completed and
there is no KFAC energy trajectory to summarize.

Interpretation: the current JAX `0.4.30` stack can run the Adam path with
neutralized tags, but the legacy DeepSolid KFAC tag/tracing path is still not
compatible with modern JAX during optimizer initialization.

## Update: KFAC Compatibility Fixed

Subsequent compatibility patches brought the real KFAC path through pretraining,
optimizer initialization, and VMC training on the current JAX stack.

- Successful job: `125926`
- Job name: `solidnes-ccpvdz-kfac-b4096-i1000-fix4`
- Partition/node: `intelgpu80g/gpu001`
- Resources: `2 GPU + 96 CPU`
- Runtime: `00:36:27`
- Final checkpoint: `qmcjax_ckpt_000999.npz`
- Training rows: `999`, steps `1--999`
- Final energy: `-75.3675107233 Ha`
- Last-100 mean energy: `-75.3358357089 Ha`
- Last-100 mean `pmove`: about `0.540`
- Minimum single-step energy: `-75.4111566672 Ha`

The 1000-step KFAC probe is therefore usable as a continuation checkpoint, but
not as a final benchmark. It still needs longer training and a separate fixed
checkpoint inference/evaluation run before reporting a production energy.

## Queued Continuation

Submitted a 10000-additional-step KFAC continuation from `qmcjax_ckpt_000999`.

- New experiment config:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000.yaml`
- New train config:
  `configs/train/ground_state_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000.yaml`
- Output directory:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results`
- Restore directory:
  `tasks/phase1_diamond_c/ccpvdz/training/0028_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000/results/checkpoints`
- Total DeepSolid iterations: `11000`
- Expected actual continuation steps: `1000--10999`
- Checkpoint interval: `30 min`
- Initial Slurm job: `126472`
- Initial job name: `solidnes-ccpvdz-kfac-cont10k-queue`
- Submission mode: `SOLIDNES_GPU_QUEUE_MODE=flexible`, i.e. the no-free-node
  queue branch rather than a pinned currently idle node
- Initial requested resources: `2 GPU`, `96 CPU`, `128000 MB`, `12:00:00`
- Partition request: `h200,amdgpu80g,amdgpu40g,h20,h800`
- Plan JSON:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/outputs/slurm_plans/deepsolid_ccpvdz_paper_kfac_continue_ckpt999_iter11000_queue_plan.json`

At submission check, Slurm reported job `126472` as `PENDING`, which is expected
for this queue-mode request.

Correction: the project GPU scheduling rule for this run is `4 GPU + 64 CPU`.
The initial `2 GPU + 96 CPU` request was cancelled and replaced.

- Cancelled job: `126472`
- Corrected Slurm job: `126502`
- Corrected job name: `solidnes-ccpvdz-kfac-cont10k-g4c64-q`
- Corrected requested resources: `4 GPU`, `64 CPU`, `128000 MB`, `12:00:00`
- Corrected plan JSON:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/outputs/slurm_plans/deepsolid_ccpvdz_paper_kfac_continue_ckpt999_iter11000_g4c64_queue_plan.json`
- Corrected submission mode: `SOLIDNES_GPU_QUEUE_MODE=flexible`
- Corrected partition request: `h200,amdgpu80g,amdgpu40g,h20,h800`
- Status at check: `PENDING`

H800 account correction on 2026-05-23: `h800` is not available to the default
`hmt03` account, so the GPU scheduling defaults now block `h800` and the
flexible queue partition set excludes it.

- Cancelled job: `126502`
- Replacement Slurm job: `126584`
- Replacement requested resources: `4 GPU`, `64 CPU`, `128000 MB`, `12:00:00`
- Replacement plan JSON:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/outputs/slurm_plans/deepsolid_ccpvdz_paper_kfac_continue_ckpt999_iter11000_g4c64_no_h800_plan.json`
- Replacement partition request: `h200,amdgpu80g,amdgpu40g,h20`
- Status at check: `PENDING`

Runtime-control correction on 2026-05-23: job `126584` started on `gpu005`
but failed after the obsolete 300-second non-Slurm process guard returned
`124:0`. The guard has been removed from the backend runner and CPU/GPU Slurm
wrappers; Slurm walltime is now the only runtime limit for submitted jobs.

- Failed partial job: `126584`
- Partial rows preserved as:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/checkpoints/train_stats_126584_partial_timeout.csv`
- Replacement Slurm job: `127816`
- Replacement job name: `solidnes-ccpvdz-kfac-cont10k-g4c64-st`
- Replacement requested resources: `4 GPU`, `64 CPU`, `128000 MB`, `12:00:00`
- Replacement plan JSON:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/outputs/slurm_plans/deepsolid_ccpvdz_paper_kfac_continue_ckpt999_iter11000_g4c64_slurm_time_only_plan.json`
- Runtime control: Slurm `--time=12:00:00` only
- Status at check: `PENDING`

The Slurm-walltime-only replacement job completed successfully.

- Completed Slurm job: `127816`
- Slurm state: `COMPLETED`, exit `0:0`
- Runtime: `03:30:43`
- Allocated partition: `amdgpu40g`
- Completed rows: `10000`, steps `1000--10999`
- Final checkpoint:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/checkpoints/qmcjax_ckpt_010999.npz`
- Summary JSON:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/validation/training_summary.json`
- Summary Markdown:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/validation/training_summary.md`
- Training evolution plot:
  `tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/validation/training_evolution.png`

Training summary:

- First energy: `-75.3626140667 Ha`
- Last energy: `-75.4010653122 Ha`
- Last-100 mean energy: `-75.4192298683 Ha`
- Tail mean, last 2000 rows: `-75.4161279970 Ha`
- Tail block stderr: `0.0006453174 Ha`
- Final variance: `0.5978799387`
- Tail variance mean: `0.6980870486`
- Mean `pmove`: `0.5397935999`
- Same-cell cc-pVDZ HF: `-74.9757591792 Ha`
- Tail mean minus HF: `-0.4403688179 Ha`
- DeepSolid paper VMC reference: `-75.4009 Ha`
- Tail mean minus paper VMC: `-0.0152279970 Ha`

The continuation is a clear improvement over the 1000-step KFAC probe:
last-100 mean moved from `-75.3358357089 Ha` to `-75.4192298683 Ha`, an
additional `83.394 mHa` decrease, and the variance dropped into the
`~0.6--0.7` tail range. The final checkpoint is suitable for a separate
fixed-checkpoint evaluation run.
