# Progress

Last updated: 2026-05-23

## Current Status

The project has been named `SolidNES`, the research-management scaffold is in
place, the first DeepSolid backend smoke path has passed on CPU plus CUDA GPU,
and a new FermiNet PBC diamond baseline scaffold has been added for the
efficiency-first route.

Current phase:

```text
Phase 0C complete through direct DeepSolid adapter-object probe
```

## Active Objective

Continue Phase 1 method integration on carbon diamond while shifting the
high-efficiency backend route toward FermiNet PBC baselines. The retired
target-specific scaffold has been removed from active project structure.

Current strategy:

```text
Adapter-first for project control; use DeepSolid for completed reference runs
and FermiNet as the new efficiency baseline before adding acceleration modules.
```

Recent additions:

- Migrated legacy artifacts into task-centered bundles under `tasks/`.
  Numbered task bundles now run through `0046`; the next available number is
  `0047`. Human-readable task outcomes are tracked in `tasks/TASK_LEDGER.md`.
- Added global agent/startup and state-tracking files: `AGENTS.md`,
  `CURRENT_STATUS.md`, and `ACTIVE_TASK.md`. Every new work session should read
  `AGENTS.md` first, then the current status/task files.
- Added numbered task-bundle tracking: new runs must preserve task-type
  grouping and use a single numbered bundle such as
  `tasks/phase1_diamond_c/pbc_gamma/training/0001_.../`. Each bundle contains
  its own `results/`, `outputs/`, and `logs/` folders so files do not pile up
  in shared top-level directories. The canonical ledger is
  `records/run_index.md`, and the rule is documented in
  `docs/00_project_guidance/result_output_numbering.md`.
- Added true PySCF PBC-HF orbital pretraining for the local FermiNet PBC route.
  The implementation follows DeepSolid's practical design: PySCF PBC KHF/KUHF
  evaluates occupied orbital matrices on the host each pretraining iteration,
  then a pmapped FermiNet pretraining step minimizes orbital-matrix MSE against
  `network.orbitals` while refreshing walkers with the neural wavefunction.
- Added the ready-to-submit PBC-HF pretrained paper pilot:
  `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot`.
  Config build passes in the FermiNet venv, and a local one-step PBC pretraining
  probe passed on CPU with a `sto-3g` target.
- Added a local shallow clone of upstream FermiNet at `external/ferminet`
  (current HEAD `c4312c315dda1c5728994ba89629744f71c6eb66`; ignored by the
  repository's `external/` rule).
- Added SolidNES FermiNet adapter support:
  `src/solidnes/backends/ferminet_adapter.py` and
  `src/solidnes/backends/ferminet_configs/diamond_pbc_gamma.py`.
- Added all-electron diamond primitive-cell PBC Gamma-point configs for
  FermiNet:
  `diamond_c_ferminet_pbc_gamma_smoke` and
  `diamond_c_ferminet_pbc_gamma_kfac_baseline`.
- Added latest-JAX CUDA environment scaffold:
  `configs/env/ferminet_jax0101_cuda12.yml`, targeting current PyPI
  `jax==0.10.1` / `jaxlib==0.10.1` with `jax[cuda12]`.
- Enabled easy first-pass speed defaults in the FermiNet route: FOLX Forward
  Laplacian (`cfg.optim.laplacian = "folx"`), KFAC for the baseline run,
  FP64 disabled through the runtime profile, GPU memory preallocation, and
  PBC feature/envelope functions.
- Added FermiNet utility scripts:
  `build_ferminet_config.py`, `check_ferminet_environment.py`,
  `run_ferminet_train.py`, and `setup_ferminet_jax_latest_env.sh`.
- Added FermiNet GPU SLURM smoke submission files:
  `run_ferminet_gpu_smoke.slurm` and `submit_ferminet_gpu_smoke.sh`.
- Conda environment creation for FermiNet was blocked by a `conda-forge`
  metadata HTTP failure, so a venv fallback was added and populated at
  `.venv/ferminet-jax0101-cuda12`.
- The venv check passes with `jax==0.10.1`, `jaxlib==0.10.1`, `folx==0.2.20`,
  `kfac_jax==0.0.8`, `pyscf==2.13.0`, and FermiNet editable install.
- Added two latest-JAX compatibility shims for the FermiNet route:
  a PBC local-energy wrapper that accepts `ndim` from upstream `train.py`, and
  a JAX 0.10 shim for `jax.device_put_replicated` /
  `jax.device_put_sharded` calls still used by kfac-jax.
- A local CPU one-step FermiNet PBC diamond smoke completed successfully in the
  venv and wrote `train_stats.csv` with one finite row:
  `energy=-16.419584 Ha`, `pmove=0.86875004`. This is only an API/runtime
  compatibility check, not a physics or performance result.
- FOLX emitted repeated warnings that `tile` is not in its registry in
  `construct_symmetric_features`, meaning the current FermiNet feature path may
  fall back to full-Hessian work for that operation. This is now a known
  Forward-Laplacian optimization target.
- Added a no-compute FermiNet CPU/config-check SLURM path:
  `run_ferminet_cpu_config_check.slurm` and
  `submit_ferminet_cpu_config_check.sh`. It defaults to `test`.
- Verified SLURM dry-runs: the no-compute config check is forced to `test`,
  while the FermiNet GPU training smoke blocks `test` by default and plans a
  one-GPU A100 job.
- Ran actual FermiNet GPU smoke as SLURM job `127828` on `intelgpu80g/gpu001`
  with `1 GPU`, not on `test`. The job started at `2026-05-23 12:28:02 CST`
  and ended at `2026-05-23 12:29:55 CST`. JAX reported
  `jax_default_backend=gpu` and `jax_devices=[CudaDevice(id=0)]`.
- The GPU smoke wrote one `train_stats.csv` row:
  `step=0`, `energy=-16.7556 Ha`, `ewmean=-16.7556`, `ewvar=-0.0`,
  `pmove=0.85625`. No traceback was present in stdout/stderr.
- The GPU smoke stderr contained 12 FOLX `tile not in registry` warnings. Per
  current work order, this warning is recorded but deferred until after the GPU
  path is considered runnable.
- Added and ran a short FermiNet PBC diamond Adam trend baseline:
  `diamond_c_ferminet_pbc_gamma_adam_short100`. It uses batch size `64`,
  `100` Adam iterations, `50` MCMC burn-in steps, MCMC10, FP32 speed profile,
  and FOLX enabled.
- Submitted the short baseline as SLURM job `127830` on `intelgpu80g/gpu001`
  with `1 GPU`, not on `test`. The job started at `2026-05-23 12:34:28 CST`
  and ended at `2026-05-23 12:36:25 CST`.
- Short-baseline output:
  `rows=100`, first energy `-21.477194 Ha`, last energy `-23.146812 Ha`,
  minimum energy `-28.307531 Ha` at step `68`, first-50 mean
  `-23.314920 Ha`, last-50 mean `-25.320289 Ha`, mean `pmove=0.901219`.
  This is a runtime/trend pass only, not a converged physics result.
- The short baseline produced 18 FOLX `tile not in registry` warnings and no
  traceback.
- Added `scripts/validation/summarize_ferminet_benchmark.py` and generated a
  benchmark summary for the short FermiNet Adam run. The summary records the
  SLURM plan, selected GPU node, runtime, energy statistics, `pmove`, FOLX
  warning count, and traceback count.
- Removed the FOLX `tile not in registry` warning in the local upstream
  FermiNet clone by replacing the repeated spin-channel mean `jnp.tile(...)`
  with `jnp.broadcast_to(...)` in `external/ferminet/ferminet/networks.py`.
  The reproducible patch is stored at
  `patches/ferminet/folx_tile_broadcast.patch`.
- Ran the after-fix comparison as SLURM job `127833` through the FermiNet GPU
  submitter. The dry-run plan blocked `test`, selected `intelgpu80g/gpu001`,
  and requested `1` A100 GPU.
- After-fix comparison output:
  `rows=100`, first energy `-19.545017 Ha`, last energy `-23.293020 Ha`,
  minimum energy `-27.982136 Ha` at step `65`, first-50 mean
  `-22.971573 Ha`, last-50 mean `-25.811617 Ha`, mean `pmove=0.903578`.
  Runtime stayed at `117 s` total, or `1.17 s/step`.
- The after-fix run reduced FOLX `tile not in registry` warnings from `18` to
  `0`; tracebacks remained `0`. This is a warning/runtime hygiene fix, not a
  convergence or physics-accuracy result.
- Added a no-FOLX ablation config:
  `diamond_c_ferminet_pbc_gamma_adam_short100_defaultlap`, using
  FermiNet's `laplacian: default` with the same 100-step Adam short setup.
- Submitted the no-FOLX comparison through the FermiNet GPU submitter as job
  `127837`. The dry-run plan blocked `test`, selected `intelgpu80g/gpu001`,
  and requested `1` A100 GPU.
- No-FOLX comparison output:
  `rows=100`, runtime `57 s`, `0.57 s/step`, first energy
  `-21.763638 Ha`, last energy `-25.978733 Ha`, last-50 mean
  `-26.592552 Ha`, mean `pmove=0.899469`, FOLX tile warnings `0`,
  tracebacks `0`.
- Forward-Laplacian ablation result for this small 12-electron, 100-step run:
  FOLX took `117 s` (`1.17 s/step`) while default Laplacian took `57 s`
  (`0.57 s/step`), so FOLX was about `2.05x` slower end-to-end at this scale.
  This includes startup/compilation overhead and should be followed by a
  larger scaling benchmark before making a general performance conclusion.
- Increased the FermiNet A/B benchmark batch size from `64` to `4096` using new
  configs rather than mutating completed-run configs:
  `diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_defaultlap` and
  `diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_folx`.
- Submitted both batch4096 jobs through the FermiNet GPU submitter. Dry-runs
  blocked `test`; both plans selected `intelgpu80g/gpu001` with `1` A100 GPU.
  Jobs: default `127840`, FOLX `127841`.
- Batch4096 comparison output:
  default Laplacian `110 s` total / `1.10 s/step`, FOLX `155 s` total /
  `1.55 s/step`, no FOLX tile warnings, no tracebacks, mean `pmove` about
  `0.90` in both runs.
- Batch scaling narrowed the FOLX overhead from `2.05x` at batch64 to `1.41x`
  at batch4096, but FOLX remained slower for the current 12-electron small
  FermiNet diamond primitive-cell setup. Current practical baseline should use
  `laplacian: default` until a larger-electron scaling test shows otherwise.
- Added and ran a FermiNet KFAC batch4096, 400-step benchmark using the current
  faster default Laplacian:
  `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_defaultlap`.
- Submitted the KFAC run through the FermiNet GPU submitter as job `127842`.
  The dry-run plan blocked `test`, selected `intelgpu80g/gpu001`, and requested
  `1` A100 GPU.
- KFAC result:
  `rows=400`, runtime `189 s`, `0.4725 s/step`, first energy
  `-20.792610 Ha`, last energy `-73.746570 Ha`, minimum energy
  `-73.897995 Ha` at step `387`, last-50 mean `-73.443896 Ha`, mean
  `pmove=0.777777`, last `ewvar=0.018472`, FOLX tile warnings `0`,
  tracebacks `0`.
- Compared with Adam batch4096/default job `127840`, KFAC reached a much lower
  energy scale quickly and was faster per logged optimization step
  (`0.4725 s/step` vs `1.10 s/step`), though this is still an early
  optimization pass rather than a converged energy.
- Added the missing matched KFAC/FOLX comparison:
  `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_folx`. This keeps
  optimizer `kfac`, batch size `4096`, and `400` iterations unchanged, changing
  only the Laplacian setting from `default` to `folx`.
- Submitted KFAC/FOLX job `127843` through the FermiNet GPU submitter. The
  dry-run plan blocked `test`, selected `intelgpu80g/gpu001`, and requested
  `1` A100 GPU.
- Matched KFAC Forward-Laplacian comparison:
  default job `127842` took `189 s` / `0.4725 s/step`; FOLX job `127843` took
  `198 s` / `0.4950 s/step`. FOLX had zero tile warnings and zero tracebacks,
  but was still about `1.0476x` slower end-to-end for the current 12-electron
  small-cell benchmark.
- Added and ran a longer KFAC/batch4096/2000-step Forward-Laplacian comparison
  to reduce initialization/JIT overhead:
  `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_defaultlap` and
  `diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_folx`.
- Submitted the 2000-step jobs through the FermiNet GPU submitter. Dry-runs
  blocked `test`; both plans selected `intelgpu80g/gpu001` with `1` A100 GPU.
  Jobs: default `127844`, FOLX `127845`.
- KFAC 2000-step result:
  default took `557 s` / `0.2785 s/step`; FOLX took `394 s` /
  `0.1970 s/step`. Both runs had zero FOLX tile warnings and zero tracebacks.
  FOLX was about `1.41x` faster after the one-time overhead was amortized.
- Updated efficiency conclusion for this setup: use `KFAC + batch4096 + FOLX`
  for longer runs. Use short default-Laplacian runs only for cheap smoke checks
  where lower initialization overhead matters more than steady-state speed.
- Added and submitted the first paper-like FermiNet comparison probe:
  `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter1000_paper`. It uses
  hidden dims `(256, 32) x 4`, 8 determinants, KFAC, FOLX, batch `4096`,
  MCMC20, and `1000` iterations.
- Submitted this probe as SLURM job `127847` through the FermiNet GPU
  submitter. The dry-run plan blocked `test`, selected
  `intelgpu80g/gpu001`, requested `--gres=gpu:2 --cpus-per-task=96`, and used
  `--exclusive`.
- Startup verification for job `127847`: `CUDA_VISIBLE_DEVICES=0,1`,
  `SOLIDNES_EFFECTIVE_CPUS=96`, `jax_default_backend=gpu`, and
  `jax_devices=[CudaDevice(id=0), CudaDevice(id=1)]`. The job started at
  `2026-05-23 14:21:34 CST` and is pending completion.
- The paper-like 1000-step probe job `127847` completed in `428 s`
  (`0.428 s/step`) with `1000` rows, last energy `-75.029800 Ha`, last-50 mean
  `-75.071278 Ha`, zero FOLX tile warnings, and zero tracebacks.
- Added and submitted the formal 10000-step paper-like FermiNet comparison run:
  `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter10000_paper`. It keeps
  the same paper-like model, KFAC, FOLX, batch `4096`, MCMC20, and 2-GPU full
  `intelgpu80g/gpu001` allocation as the probe.
- Submitted job `127848` through the FermiNet GPU submitter. The dry-run plan
  blocked `test`, selected `intelgpu80g/gpu001`, requested
  `--gres=gpu:2 --cpus-per-task=96`, and used `--exclusive`.
- Startup verification for job `127848`: `CUDA_VISIBLE_DEVICES=0,1`,
  `SOLIDNES_EFFECTIVE_CPUS=96`, `jax_default_backend=gpu`, and
  `jax_devices=[CudaDevice(id=0), CudaDevice(id=1)]`. The job started at
  `2026-05-23 14:37:33 CST`.
- The formal 10000-step FermiNet comparison job `127848` completed successfully
  in `1748 s` by log timestamps (`00:29:16` by `sacct`) with `10000` rows and
  zero tracebacks. It used `2 GPU + 96 CPU` on `intelgpu80g/gpu001`, KFAC,
  FOLX Forward Laplacian, batch `4096`, MCMC20, hidden dims `(256, 32) x 4`,
  and 8 determinants.
- FermiNet 10k result:
  last energy `-75.281360 Ha`, last-50 mean `-75.26011696 Ha`, last-5000 tail
  mean `-75.2595080136 Ha`, tail block stderr `0.0027959247 Ha`, mean
  `pmove=0.5538866356`, last `ewvar=0.0017481738`, and zero FOLX tile
  warnings.
- Compared with the existing DeepSolid cc-pVDZ 150000-step continuation
  reference job `122085` on the same 2-GPU A100 node class, FermiNet reached a
  lower tail estimate in about `12.04x` less wall-clock time: FermiNet tail
  mean `-75.2595080136 Ha` versus DeepSolid tail mean `-75.1982415149 Ha`.
  FermiNet last energy was also lower than the DeepSolid final energy by
  `0.0108978656 Ha`. This makes FermiNet KFAC/FOLX the current efficiency
  baseline for the project.
- Added and submitted an x64 paper-geometry FermiNet pilot:
  `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot`.
  It uses the DeepSolid supplementary diamond geometry (`a = 3.5738 Angstrom`,
  primitive vector components `1.7869 Angstrom`), KFAC, FOLX, batch `4096`,
  burn-in `1000`, MCMC20, `20000` iterations, and `JAX_ENABLE_X64=1`.
- The x64 pilot intentionally does not enable `pretrain_method: hf` because
  upstream FermiNet's pretraining path targets molecular PySCF orbitals, not a
  faithful PBC-HF target for the current periodic feature/envelope route.
  Implementing true PBC-HF cc-pVDZ pretraining remains the next accuracy task if
  x64 and paper geometry alone do not move the energy enough.
- Submitted the x64 pilot through `scripts/slurm/submit_ferminet_gpu_smoke.sh`
  as job `127898`. Dry-run blocked `test`, selected `intelgpu80g/gpu001`, and
  requested `--gres=gpu:2 --cpus-per-task=96 --exclusive`. Startup verification
  showed `CUDA_VISIBLE_DEVICES=0,1`, `JAX_ENABLE_X64=1`,
  `jax_default_backend=gpu`, and `jax_devices=[CudaDevice(id=0), CudaDevice(id=1)]`.
- The x64 pilot job `127898` completed successfully in `8579 s` (`02:23:08`
  by `sacct`) with `20000` rows, zero FOLX tile warnings, and zero tracebacks.
  Result: last energy `-75.4007786448 Ha`, last-50 mean `-75.4161488268 Ha`,
  last-1000 mean `-75.4142544132 Ha`, last-10000 tail mean
  `-75.4114059555 Ha`, tail block stderr `0.0010328659 Ha`, mean
  `pmove=0.5446185901`, and last `ewvar=0.0001143532`.
- Compared with the DeepSolid supplementary diamond value `-75.4009(2) Ha`,
  the FermiNet x64 pilot tail estimates are lower by roughly `0.0105-0.0134 Ha`.
  This is a strong training result but not yet a final publishable energy,
  because it is still based on training-chain statistics rather than an
  independent fixed-parameter inference run.
- Added and submitted a quick fixed-parameter FermiNet evaluation pilot:
  `diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000`.
  It restores the latest saved x64 training checkpoint `qmcjax_ckpt_018349.npz`,
  uses `optimizer: none`, x64, FOLX, batch `4096`, MCMC20, burn-in `0`, and
  `2000` evaluation iterations. Submitted as job `127992` through the GPU
  submitter; dry-run blocked `test` and selected `intelgpu80g/gpu001` with
  `2 GPU + 96 CPU`.
- Job `127992` completed successfully in `00:22:29` with `2000` fixed-parameter
  evaluation rows, zero FOLX tile warnings, and zero tracebacks. The all-row
  energy mean is `-75.4125655570 Ha`, 5-block stderr `0.0004411545 Ha`,
  last-1000 mean `-75.4118625314 Ha`, last energy `-75.4374770841 Ha`,
  mean `pmove=0.5362353088`, and last `ewvar=0.0001866115`. This agrees with
  the x64 training tail-10000 mean `-75.4114059555 Ha` within about
  `0.00116 Ha`, so the fixed-checkpoint evaluation confirms the trained
  wavefunction estimate rather than relying only on training-chain statistics.
  Plots were written under
  `tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/`.
- Configured and submitted a short paper-like C-diamond cc-pVDZ KFAC probe:
  `diamond_c_deepsolid_ccpvdz_paper_kfac_batch4096_mcmc20_iter1000`.
  The config uses `(256, 32) x 4`, 8 determinants, KFAC learning rate `3e-2`,
  batch size `4096`, pretrain `1000`, burn-in `1000`, MCMC20, and 1000 VMC
  iterations. During submission, two compatibility issues were fixed: GPU
  auto-scheduling now accepts allowed GPU counts so `4096` is not paired with
  invalid 3-GPU requests, and DeepSolid pretraining temporarily neutralizes
  legacy KFAC tags before restoring real tags for KFAC optimization. Active
  SLURM job `125574` ran on `intelgpu80g/gpu001`, `2 GPU + 96 CPU`, exclusive,
  but failed after `00:07:10` during `optimizer.init(...)` for KFAC with
  `TypeError: cannot unpack non-iterable ShapedArray object` in the legacy
  KFAC tag path. HF converged at `-74.9757591791533 Ha`, a post-pretrain
  checkpoint `qmcjax_ckpt_000000.npz` was written, but no `train_stats` file
  was produced and no KFAC VMC iterations completed.
- Implementation strategy document.
- VMC_reproduce reference notes.
- Decision record 0002.
- Role-split Phase 1 config scaffold.
- Backend survey and decision record 0003.
- DeepSolid environment plan and decision record 0004.
- Carbon diamond DeepSolid smoke-test scaffold.
- DeepSolid legacy environment creation attempt record.
- Decision record 0005 and newer JAX probe environment scaffold.
- Working `solidnes-deepsolid-jax0430-probe` environment for import/config smoke.
- DeepSolid config builder and two-layer diamond smoke strategy.
- Passing zero-iteration DeepSolid runtime smoke.
- Decision record 0007 for smoke-only DeepSolid compatibility shims.
- DeepSolid adapter interface guidance document.
- FIIR-style SLURM scheduling scaffold for CPU/GPU jobs; CPU/GPU dry-run plans
  pass.
- Actual CPU SLURM smoke passed on job `120615`.
- Actual GPU SLURM smoke submitted as job `120612`, but failed because the
  current JAX environment is CPU-only.
- Created `solidnes-deepsolid-jax0430-cuda12-probe` from
  `configs/env/deepsolid_py39_jax0430_cuda12_probe.yml`.
- Actual CUDA GPU SLURM smoke passed on job `120634` using `gpu4090_128` /
  `gpu40903`; JAX reported `jax_default_backend=gpu` and `jax_devices=[cuda(id=0)]`.
- Decision record 0009: DeepSolid remains the first backend for Phase 1
  adapter work.
- Added `src/solidnes/backends/deepsolid_adapter.py`; DeepSolid scripts are now
  thin wrappers around the adapter.
- Added one-step Adam smoke configs.
- First one-step GPU attempt `120654` reached training stats but failed during
  historical DeepSolid checkpoint saving.
- Added smoke-only object-array checkpoint save shim.
- One-step Adam GPU retry `120655` passed on `gpu4090_128` / `gpu40903` with
  `COMPLETED 0:0`.
- Added `DeepSolidGroundStateObjects` and
  `initialize_deepsolid_ground_state(...)` to expose initialized DeepSolid
  network apply functions, params, walkers, local energy, and MCMC transition.
- Added `scripts/backends/probe_deepsolid_adapter_objects.py`.
- Direct adapter-object probe passed locally in
  `solidnes-deepsolid-jax0430-probe` for carbon diamond:
  `basis=sto-3g`, `pseudo=None`, `nelectron=12`, `data_shape=(1, 8, 36)`,
  finite local energy/variance, and `pmove=1`.
- Direct adapter-object probe also passed on CUDA GPU as SLURM job `120686`
  using `gpu4090_128` / `gpu40904`; `jax_default_backend=gpu`,
  `jax_devices=[cuda(id=0)]`, `COMPLETED 0:0`.
- SLURM DeepSolid CPU/GPU templates now accept `SOLIDNES_BACKEND_SCRIPT`, so
  the same scheduling wrapper can run either process smoke or adapter probes.
- Added carbon-diamond validation configs and scripts:
  `compute_pyscf_pbc_reference.py` and `summarize_deepsolid_validation.py`.
- Computed the same-cell PySCF PBC KHF reference for carbon diamond `sto-3g`:
  `E_HF = -74.0041967316 Ha`, converged.
- Added a paper-alignment `ccpvdz` HF reference config for carbon diamond:
  `diamond_c_deepsolid_ccpvdz_hf_reference`. The PySCF PBC KHF reference
  converged at `E_HF = -74.9757591792 Ha`, about `+0.0026408208 Ha` above the
  DeepSolid paper small-cell HF value `-74.9784 Ha`.
- Submitted the first carbon `ccpvdz` neural-network VMC sanity run:
  `diamond_c_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity`.
  It uses `(256, 32) x 4`, 8 determinants, batch size 384, 1000 HF-target
  pretrain iterations, and 1000 Adam VMC iterations. SLURM job `121231` started
  on `gpu4090_128` / `gpu40903` with 8 RTX 4090 GPUs.
- Job `121231` completed successfully in `00:09:27`. The run wrote 1000
  training rows and checkpoint `qmcjax_ckpt_000999.npz`. Energy decreased from
  `-70.8880 Ha` to `-72.8789 Ha`, variance decreased from `187.6560` to
  `38.6163`, and mean `pmove=0.5184`. The last energy remains
  `+2.0969 Ha` above the same-cell `ccpvdz` HF reference, so this is a
  stability/trend pass rather than an accuracy pass.
- Submitted cc-pVDZ paper-net continuation job `121253` to continue from
  `qmcjax_ckpt_000999` to 10000 total VMC steps. It uses Adam `2e-4`,
  batch size 384, MCMC 20, and runs on `gpu4090_128` / `gpu40903` with
  8 RTX 4090 GPUs.
- Job `121253` completed successfully in `00:52:16`, writing 9000 training rows
  and checkpoint `qmcjax_ckpt_009999.npz`. Energy decreased from
  `-72.8129 Ha` at step 1000 to `-74.6359 Ha` at step 9999; variance decreased
  from `40.1310` to `23.2866`, mean `pmove=0.5383`, and the last-step gap to
  same-cell `ccpvdz` HF is now `+0.3399 Ha`. The last-half tail mean remains
  `+0.6823 Ha` above HF and block means are still drifting downward, so this is
  a strong progress result but not convergence.
- Submitted cc-pVDZ paper-net continuation job `121338` to continue from
  `qmcjax_ckpt_009999` to 30000 total VMC steps. It uses Adam `1e-4`,
  batch size 384, MCMC 20, and runs on `gpu4090_128` / `gpu40903` with
  8 RTX 4090 GPUs.
- Job `121338` completed successfully in `01:55:56`, writing 20000 training
  rows and checkpoint `qmcjax_ckpt_029999.npz`. Energy decreased from
  `-74.3952 Ha` at step 10000 to `-74.7065 Ha` at step 29999; variance
  decreased from `21.3085` to `15.2261`, mean `pmove=0.5392`, and the
  last-half tail mean is `-74.9086 Ha`, only `+0.0671 Ha` above same-cell
  `ccpvdz` HF. The final tail block mean is `-74.9583 Ha`, about `+0.0175 Ha`
  above HF.
- Submitted cc-pVDZ paper-net continuation to continue from
  `qmcjax_ckpt_029999` to 50000 total VMC steps. The initial 8-GPU job
  `121397` and 4-GPU job `121398` were both canceled before start. Active job
  `121423` follows the current GPU submission policy via
  `scripts/slurm/submit_deepsolid_gpu_smoke.sh`: Adam `5e-5`, batch size 384,
  MCMC 20, `2 GPU`, exclusive `96 CPU`, `16:00:00`, running on
  `intelgpu80g` / `gpu001` as `best_idle_gpu_node`.
- Job `121423` completed successfully in `01:10:36`, writing 20000 training
  rows and checkpoint `qmcjax_ckpt_049999.npz`. Energy decreased from
  `-74.8393 Ha` at step 30000 to `-75.2078 Ha` at step 49999; variance
  decreased from `10.2293` to `6.7056`, mean `pmove=0.5394`, and the last-half
  tail mean is `-75.0910 Ha`, `-0.1152 Ha` below the same-cell `ccpvdz` HF
  reference. The five tail block means end nearly flat around `-75.107 Ha`.
- Submitted cc-pVDZ paper-net continuation job `122085` to continue from
  `qmcjax_ckpt_049999` to 150000 total VMC steps, adding another 100000 Adam
  steps at learning rate `2e-5`. It uses the current GPU auto-sizing policy
  and started on `intelgpu80g` / `gpu001` with `2 GPU`, exclusive `96 CPU`,
  and `16:00:00` walltime.
- Job `122085` completed successfully in `05:50:50`, writing 100000 training
  rows and checkpoint `qmcjax_ckpt_149999.npz`. The last-half tail mean is
  `-75.1982415149 Ha`, an improvement of `0.107247 Ha` over the 50000-step
  tail mean `-75.0909943902 Ha`, and remains `+0.202658 Ha` above the
  DeepSolid paper small-cell VMC value `-75.4009 Ha`. The final two tail block
  means are flat at about `-75.2045 Ha`.
- Updated the GPU scheduling policy to auto-size from current available nodes:
  the planner now separates hard minimum GPUs from target/preferred GPUs and
  derives `--gres=gpu:N` from the selected node before falling back to flexible
  queueing. The submitter treats legacy `SOLIDNES_GPU_MIN_GPUS` as a target
  count and adds `SOLIDNES_GPU_HARD_MIN_GPUS` for the true lower bound.
- Ran 20-step GPU validation job `120704`; finite stats and useful MCMC
  acceptance, but energy trended upward, so this was not an accuracy pass.
- Ran lower-learning-rate 50-step stable GPU validation job `120710`;
  energy improved from `-28.2426 Ha` to `-41.5337 Ha`, minimum
  `-49.0733 Ha`, variance decreased, and mean `pmove=0.5425`.
- Current validation conclusion: the accuracy-check harness works, but the
  current small `sto-3g` detnet/short Adam runs are not accurate or converged;
  the best validation energy is still `24.9308 Ha` above HF.
- Added `diamond_c_deepsolid_validation_pretrain` with HF-target `net`
  pretraining, 100 pretrain iterations, and 100 Adam iterations.
- Ran pretrain validation job `120722` on `intelgpu80g` / `gpu001`;
  `COMPLETED 0:0`, 2 GPUs, 1m40s. Energy improved from `-32.7958 Ha` to
  `-41.9058 Ha`, minimum `-49.8937 Ha`, tail mean `-38.1651 +/- 0.9196 Ha`
  by 5-block estimate, and mean `pmove=0.5315`.
- Pretraining slightly improved the best HF gap from `24.9308 Ha` to
  `24.1105 Ha`, but the result remains far from the converged HF baseline.
- Added checkpoint restore support to the DeepSolid smoke compatibility shim.
  Validation continuation restores walkers and parameters while resetting Adam
  optimizer state.
- Ran continuation probe job `120729` from the small pretrain checkpoint to
  step 199; `COMPLETED 0:0`. Tail mean improved to `-42.9822 Ha`, with
  `tail_mean_minus_hf = +31.0219 Ha`.
- Added `deepsolid_detnet_medium` and ran medium-detnet validation job `120735`;
  `COMPLETED 0:0`, 2 A100 GPUs, 500 training rows. Tail mean improved to
  `-47.1336 Ha`, with `tail_mean_minus_hf = +26.8706 Ha`, but variance rose
  and the run is still not accurate/converged.
- Continued the medium-detnet validation from checkpoint `000499` to step
  `999` as SLURM job `120739`; `COMPLETED 0:0`, 2 A100 GPUs, 1000 total
  training rows. Tail mean improved to `-51.3345 Ha`, with
  `tail_mean_minus_hf = +22.6697 Ha`; this is a clear optimization improvement
  but remains far from the same-cell HF reference and is not an accuracy pass.
- Designed a larger about-20-minute carbon validation task:
  `diamond_c_deepsolid_validation_batch64_mcmc10_20min`, with batch size 64,
  MCMC steps per iteration 10, 300 pretrain iterations, 800 VMC iterations, and
  learning rate `0.0005`. Submitted as SLURM job `120741` on `intelgpu80g` /
  `gpu001` with 2 A100 GPUs and `00:25:00` walltime; `COMPLETED 0:0` in
  `00:02:21`. The run produced 800 training rows and checkpoint `000799`.
  Tail mean improved to `-59.0591 Ha`, with `tail_mean_minus_hf =
  +14.9451 Ha`; batch64/MCMC10 lowered rolling energy noise relative to the
  batch16 medium run, but the run is still not converged or accurate.
- Added an iteration-dominant next-run config:
  `diamond_c_deepsolid_validation_iter5000_batch96_mcmc12`, with 5000 VMC
  iterations, batch size 96, MCMC steps per iteration 12, 500 pretrain
  iterations, and learning rate `0.0003`. DeepSolid config build and SLURM
  dry-run passed for 2 A100 GPUs, 16 CPU cores, and `00:45:00` walltime.
  Submitted as SLURM job `120743` on `intelgpu80g` / `gpu001`. The SLURM job
  failed with exit `124:0` after `00:05:12` because the old non-SLURM
  runtime guard was still active, not because of physics/runtime failure. That
  guard has since been removed so Slurm walltime is the only runtime control.
  The partial run produced 3197
  training rows through step 3196; tail mean improved to `-70.8751 Ha`, with
  `tail_mean_minus_hf = +3.1291 Ha`. No final step checkpoint was saved because
  the wrapper exited before clean process completion.
- Updated GPU SLURM CPU allocation plumbing so allocated CPU cores are grouped
  by GPU and exported to runtime scripts. The GPU run script now computes and
  logs `SOLIDNES_CPU_BATCHES`, `SOLIDNES_CPUS_PER_GPU_LIST`, and uses the
  allocated CPU budget for OpenMP/MKL/OpenBLAS/NumExpr thread defaults instead
  of forcing CPU thread pools to 1.
- Ran the full `diamond_c_deepsolid_validation_iter5000_batch96_mcmc12_full`
  validation as SLURM job `120746`; `COMPLETED 0:0`, 5000 rows through step
  4999, final checkpoint `qmcjax_ckpt_004999.npz`. Tail mean is
  `-71.2519 Ha`, `tail_mean_minus_hf = +2.7523 Ha`; last-step gap to HF is
  `+1.5935 Ha`. The run still has outliers and is not a physical benchmark,
  but it is the strongest carbon `sto-3g` convergence result so far.
- Added independent fixed-checkpoint evaluation support through
  `output.restore_checkpoint_dir`, then configured
  `diamond_c_deepsolid_evaluate_ckpt4999_batch96_mcmc20`. This run restores
  `qmcjax_ckpt_004999.npz`, keeps network parameters fixed with
  `optimizer: none`, and evaluates steps 5000--9999 with MCMC20 to estimate
  the trained wavefunction energy more cleanly. Submitted as SLURM job
  `120750` on `gpu004` with 4 A100 GPUs and 64 CPU cores; runtime logs confirm
  CPU batches `gpu0:0-15;gpu1:16-31;gpu2:32-47;gpu3:48-63`.
- Job `120750` completed successfully in `00:06:09` and produced 5000 fixed
  checkpoint evaluation rows. The all-sample energy mean is
  `-72.0377 Ha`, with block stderr `0.0217 Ha` and gap to the same-cell HF
  reference `+1.9665 Ha`. The second half mean is `-71.9957 Ha`, and the last
  1000-row mean is `-72.0124 Ha`, so the fixed-wavefunction estimate is stable
  around `-72.0 Ha`. Mean `pmove = 0.5164` is healthy. The result is a cleaner
  estimate than the noisy training tail, but it is still about 2 Ha above HF and
  should not be called accurate.
- Configured the next optimization diagnostic:
  `diamond_c_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20`. It restores the
  same `qmcjax_ckpt_004999.npz`, continues Adam optimization from step 5000 to
  9999, uses `learning_rate = 2e-4` with delay `5000`, MCMC20, batch size 96,
  and no pretraining. Submitted as SLURM job `120759` to the A100 partitions
  `amdgpu40g,amdgpu80g`; initial status is `PENDING (Resources)` for
  `gpu:4`.
- Per user request to avoid queueing, canceled pending job `120759` and
  retargeted the continuation to the currently available high-resource node
  `gpu40904`. Added checkpoint restore resharding so a checkpoint saved on 4
  GPUs can be restored on a different local GPU count. Submitted direct SLURM
  job `120760` with `gpu:6`, `96 CPU`, and `04:00:00` walltime on
  `gpu4090_128/gpu40904`; it started immediately and is `RUNNING`.
- Job `120760` completed successfully in `00:08:32` on 6 RTX 4090 GPUs and
  produced 5000 continuation rows through step `9999`, plus checkpoint
  `qmcjax_ckpt_009999.npz`. The continuation clearly improved the medium
  wavefunction: all-run mean `-72.7885 Ha`, second-half mean `-73.0953 Ha`,
  last-1000 mean `-73.2520 Ha`, last-500 mean `-73.3111 Ha`, and final step
  `-73.9764 Ha` (`+0.0278 Ha` above the same-cell HF reference). Mean `pmove`
  was `0.5274`, and mean variance dropped to `57.2864`. The run is still
  trending downward at the end, so the next decision should be based on a
  fixed-checkpoint evaluation of `qmcjax_ckpt_009999.npz`.
- Per user request, configured and submitted an additional 20000-iteration
  continuation from `qmcjax_ckpt_009999.npz`:
  `diamond_c_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000`.
  It restores the step-9999 checkpoint, trains steps `10000--29999`, uses
  `learning_rate = 1e-4`, MCMC20, batch size 96, and no pretraining. Submitted
  as SLURM job `120805` directly on `gpu40904` with `6 x RTX 4090`, `96 CPU`,
  and `04:00:00` walltime. Startup logs confirm six visible GPUs, CPU batches
  of 16 cores per GPU, and GPU utilization around `91--94%`.
- Job `120805` completed successfully in `00:27:36` and produced 20000 rows
  through step `29999`, plus checkpoint `qmcjax_ckpt_029999.npz`. The long
  continuation substantially improved the mean energy: all-run mean
  `-73.7927 Ha`, first-half mean `-73.6274 Ha`, second-half mean
  `-73.9581 Ha`, last-5000 mean `-74.0146 Ha`, last-2000 mean
  `-74.0283 Ha`, and last-1000 mean `-74.0321 Ha`. Relative to the same-cell
  HF reference `-74.0041967316 Ha`, the second-half mean is only `+0.0461 Ha`
  high, while shorter tail means are slightly below HF; this must be interpreted
  as noisy training estimates until a fixed-checkpoint evaluation of
  `qmcjax_ckpt_029999.npz` is run. Mean `pmove = 0.5313`; mean variance dropped
  further to `33.6250`.
- Started the required fixed-checkpoint validation of `qmcjax_ckpt_029999.npz`:
  `diamond_c_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000`. It uses
  `optimizer: none`, MCMC30, batch size 96, and `iterations = 40000`, so after
  restoring checkpoint step `29999` it should evaluate steps `30000--39999`
  for 10000 fixed-parameter samples. Submitted as SLURM job `120884` directly
  on `gpu40904` with `6 x RTX 4090`, `96 CPU`, and `04:00:00` walltime. Startup
  logs confirm six visible GPUs and GPU utilization around `83--85%`.
- Job `120884` completed successfully in `00:15:48` and produced 10000
  fixed-checkpoint evaluation rows through step `39999`. The independent
  fixed-wavefunction estimate confirms the checkpoint is near the same-cell HF
  reference: all-sample mean `-74.0217682184 Ha`, 10-block stderr
  `0.0069329239 Ha`, and gap to `E_HF = -74.0041967316 Ha` of
  `-0.0175714868 Ha`. First-half mean is `-74.0173872252 Ha`; second-half
  mean is `-74.0261492117 Ha`; mean `pmove = 0.5376923958`; mean variance
  `24.3897387939`. Treat this as a strong workflow validation for the current
  `sto-3g` primitive-cell setup, with the caveat that this is not a production
  physical benchmark.
- Added explicit seed plumbing for DeepSolid runs: SolidNES now maps
  `training.seed` to `cfg.debug.seed`, records `cfg.debug.params_seed`, and the
  vendored DeepSolid process entrypoint honors those optional seed fields
  instead of only using hard-coded deterministic seeds. Submitted a repeat
  fixed-checkpoint evaluation with sampler seed `20260522`:
  `diamond_c_deepsolid_evaluate_ckpt29999_seed20260522_batch96_mcmc30_iter40000`.
  It restores `qmcjax_ckpt_029999.npz`, uses `optimizer: none`, MCMC30, batch
  size 96, and evaluates steps `30000--39999` for 10000 fixed-parameter
  samples. SLURM job `120977` was submitted directly to `gpu40904` with
  `6 x RTX 4090`, `96 CPU`, and `02:00:00` walltime; startup logs confirm six
  visible GPUs and CPU batches of 16 cores per GPU.
- Job `120977` completed successfully in `00:15:55` and produced 10000
  seed-repeat fixed-checkpoint evaluation rows. The all-sample mean is
  `-74.0379473293 Ha`, with 10-block stderr `0.0071853424 Ha` and a gap to
  `E_HF = -74.0041967316 Ha` of `-0.0337505977 Ha`. First-half mean is
  `-74.0300212567 Ha`; second-half mean is `-74.0458734019 Ha`; last-1000
  mean is `-74.0053142973 Ha`; mean `pmove = 0.5370459028`; mean variance
  `25.2250710039`. Compared with the previous fixed evaluation mean
  `-74.0217682184 Ha`, the new seed is lower by `-0.0161791109 Ha`, about
  `1.62` combined block standard errors. The equal-weight mean of the two
  fixed evaluations is `-74.0298577739 Ha`, or `-0.0256610423 Ha` relative to
  HF, with an estimated mean stderr of `0.0049923586 Ha`.

## Next Actions

1. The seed-repeat fixed evaluation agrees with the previous fixed evaluation
   at the current statistical precision. The next physics lever should be a
   larger basis and/or supercell; use longer fixed evaluations only if the
   immediate goal is reducing the current `10--20 mHa` residual Monte Carlo
   uncertainty.
2. Treat partial `solidnes-deepsolid-legacy` as broken until removed/rebuilt.
3. Build the SolidNES-side backend state interface on top of the exposed carbon
   diamond DeepSolid objects: `log_psi`, params, walkers, local energy, MCMC
   step, RNG state, and scalar diagnostics.
4. Stabilize the medium-detnet validation before claiming accuracy:
   run the batch64/MCMC10 control experiment, continue with lower learning
   rates, add checkpoint walker resize before changing batch size on restored
   runs, and keep stricter tail/block criteria. Single-step minima are too
   noisy to use as success criteria.
5. Add a small carbon-diamond two-state/NES objective scaffold that can call the
   ground-state primitives without claiming a physical excitation benchmark.
6. Keep new task artifacts under numbered `tasks/.../NNNN_slug/` bundles and
   update `tasks/TASK_LEDGER.md` on completion.

## Open Questions

- Should the first ansatz be FermiNet-style or Psiformer-style?
- What compute resource is available for the first MVP run?
- Can the chosen backend be called as a library, or must it be forked?
- What diagnostics must be available before starting NES-VMC?
