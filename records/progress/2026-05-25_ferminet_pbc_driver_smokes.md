# FermiNet PBC Excited-State Driver Smokes

Date: 2026-05-25

## Summary

Promoted the smoke-validated fixed-sample paper-tangent update into a reusable
sampler-integrated FermiNet PBC excited-state driver and validated it on
scheduled GPUs with real local-energy/Laplacian evaluations.

## Source Changes

- Added `src/solidnes/excited_states/ferminet_pbc_driver.py`.
- Exported the driver helpers from `src/solidnes/excited_states/__init__.py`.
- Added `scripts/validation/check_ferminet_pbc_driver_smoke.py`.
- Updated `scripts/slurm/run_ferminet_gpu_smoke.slurm` to pass optional
  `SOLIDNES_BACKEND_ARGS` through to backend scripts.
- Updated validation docs with the driver smoke command.

The driver owns per-state random-walk Metropolis updates, PBC position wrapping,
paper-tangent guarded parameter updates, persistent diagnostics, and checkpoint
save/load helpers.

## Validation

Local checks:

```text
python -m py_compile scripts/validation/check_ferminet_pbc_driver_smoke.py src/solidnes/excited_states/ferminet_pbc_driver.py src/solidnes/excited_states/__init__.py
bash -n scripts/slurm/run_ferminet_gpu_smoke.slurm scripts/slurm/submit_ferminet_gpu_smoke.sh
git diff --check
```

The CPU cheap-local-energy driver smoke passed for both the two-walker and
four-walker experiment configs.

Scheduled GPU checks:

- Run `0068`, job `128751`: completed on `intelgpu80g/gpu001` in `00:05:44`,
  exit `0:0`, using `gpu:2` and 96 CPU cores. Walkers/state `2`; sampler
  acceptance `0.75`, `0.75`; guarded updates accepted; checkpoint roundtrip
  `1233826` bytes.
- Run `0069`, job `128752`: completed on `intelgpu80g/gpu001` in `00:04:31`,
  exit `0:0`, using `gpu:2` and 96 CPU cores. Walkers/state `4`; sampler
  acceptance `0.791667`, `0.75`; guarded updates accepted; checkpoint roundtrip
  `1234734` bytes.

## Next Work

The next useful step is to turn this smoke driver into a production runner:
persist checkpoints at configured intervals, resume from a previous checkpoint,
and then run a short controlled NES-VMC trajectory with enough iterations to
inspect overlap and state-energy drift rather than only executability.
