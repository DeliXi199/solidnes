# 2026-05-23 FermiNet PBC Diamond Scaffold

Added the first FermiNet-based framework pieces for diamond energy
reproduction:

- Local upstream FermiNet clone in `external/ferminet` at HEAD
  `c4312c315dda1c5728994ba89629744f71c6eb66`.
- SolidNES adapter:
  `src/solidnes/backends/ferminet_adapter.py`.
- Diamond PBC Gamma-point config module:
  `src/solidnes/backends/ferminet_configs/diamond_pbc_gamma.py`.
- Smoke and KFAC baseline experiment YAMLs under `configs/experiment/`.
- Latest-JAX CUDA environment file targeting `jax==0.10.1`.
- Environment check, config-build, train wrapper, and GPU SLURM submission
  scripts.

Validation completed in the current base environment:

- Python syntax compile passed for the new Python files.
- YAML parse check passed for all configs.
- FermiNet GPU SLURM dry-run produced a ready one-GPU A100 plan.
- Conda environment creation failed at `conda-forge` metadata retrieval, so a
  venv fallback was added.
- `.venv/ferminet-jax0101-cuda12` was created successfully with
  `jax==0.10.1`, `jaxlib==0.10.1`, FOLX, kfac-jax, PySCF, and editable
  FermiNet.
- Local CPU one-step smoke completed and wrote one finite stats row:
  `energy=-16.419584 Ha`, `pmove=0.86875004`.
- Added compatibility wrappers for FermiNet PBC local energy (`ndim` kwarg)
  and latest-JAX device replication APIs used by kfac-jax.
- Added a no-compute CPU config-check SLURM path that defaults to `test`.
- Confirmed GPU training smoke dry-run blocks `test` and chooses an A100
  partition.
- Submitted actual GPU smoke as job `127828`; it ran on `intelgpu80g/gpu001`
  with `1 GPU`, outside `test`.
- Job `127828` completed the FermiNet smoke wrapper. The log shows
  `jax_default_backend=gpu`, `jax_devices=[CudaDevice(id=0)]`, and
  `FermiNet GPU smoke ends at Sat May 23 12:29:55 CST 2026`.
- The smoke wrote `train_stats.csv` with:
  `step=0`, `energy=-16.7556 Ha`, `pmove=0.85625`.
- Stderr contained 12 FOLX `tile not in registry` warnings and no traceback.
- Added short trend configs:
  `metropolis_ferminet_pbc_short`,
  `ground_state_ferminet_pbc_adam_short100`, and
  `diamond_c_ferminet_pbc_gamma_adam_short100`.
- Submitted short trend job `127830`; it ran on `intelgpu80g/gpu001` with
  `1 GPU`, outside `test`, from `2026-05-23 12:34:28 CST` to
  `2026-05-23 12:36:25 CST`.
- Short trend job wrote 100 stats rows. Summary:
  first energy `-21.477194 Ha`, last energy `-23.146812 Ha`, minimum energy
  `-28.307531 Ha` at step `68`, first-50 mean `-23.314920 Ha`, last-50 mean
  `-25.320289 Ha`, mean `pmove=0.901219`.
- The short trend run had 18 FOLX `tile not in registry` warnings and no
  traceback.

Not yet completed:

- Long KFAC baseline energy reproduction.

Known issue:

- FOLX warns that `tile` is not in its registry for FermiNet symmetric feature
  construction, so current Forward Laplacian use may still fall back to
  full-Hessian work for that operation.
