# 2026-05-23 FermiNet Adam Short100

Ran the first non-test short FermiNet PBC diamond trend baseline.

Config:

- Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100`
- Model: small FermiNet PBC, 8 determinants, hidden dims `(128, 32) x 3`
- Train: Adam, 100 iterations, batch size 64, learning rate `2e-4`
- Sampler: burn-in 50, MCMC10
- Runtime: `jax==0.10.1`, FP64 off, FOLX enabled

SLURM:

- Job: `127830`
- Partition/node: `intelgpu80g/gpu001`
- GPU: 1 A100
- Not submitted to `test`
- Start/end: `2026-05-23 12:34:28 CST` to `2026-05-23 12:36:25 CST`

Output:

- Stats rows: 100
- First energy: `-21.477194 Ha`
- Last energy: `-23.146812 Ha`
- Minimum energy: `-28.307531 Ha` at step 68
- First-10 mean: `-20.914188 Ha`
- Last-10 mean: `-24.188218 Ha`
- First-50 mean: `-23.314920 Ha`
- Last-50 mean: `-25.320289 Ha`
- Mean `pmove`: `0.901219`
- Last `ewvar`: `0.796260`

Assessment:

The GPU short baseline is a runtime/trend pass. Energy statistics improved over
the first half of the run, and `pmove` stayed high but acceptable for a smoke
baseline. The last few instantaneous energies drift upward, so this is not a
convergence result.

Known issue:

- FOLX emitted 18 `tile not in registry` warnings.

Benchmark summary:

- Added a benchmark summary artifact for this run at
  `tasks/phase1_diamond_c/pbc_gamma/training/0033_ferminet_adam_short100/results/validation/benchmark_summary.json`.
- The summary records the SLURM plan, selected partition/node, runtime,
  energy statistics, `pmove`, FOLX warning count, and traceback count.

Follow-up FOLX tile fix:

- Patched the local upstream FermiNet clone so the repeated spin-channel mean
  feature uses `jnp.broadcast_to(...)` instead of `jnp.tile(...)` in
  `external/ferminet/ferminet/networks.py`.
- Captured the patch in
  `patches/ferminet/folx_tile_broadcast.patch` because `external/` is ignored.
- Submitted the same 100-step Adam comparison through the GPU scheduler path as
  job `127833`; dry-run selected `intelgpu80g/gpu001`, blocked `test`, and used
  `1` A100 GPU.
- The after-fix run completed from `2026-05-23 12:49:21 CST` to
  `2026-05-23 12:51:18 CST`.
- After-fix output: `rows=100`, first energy `-19.545017 Ha`, last energy
  `-23.293020 Ha`, minimum energy `-27.982136 Ha` at step `65`, first-50 mean
  `-22.971573 Ha`, last-50 mean `-25.811617 Ha`, mean `pmove=0.903578`.
- Runtime stayed effectively unchanged at `117 s` total and `1.17 s/step`.
- FOLX `tile not in registry` warnings dropped from `18` to `0`; tracebacks
  stayed at `0`.
