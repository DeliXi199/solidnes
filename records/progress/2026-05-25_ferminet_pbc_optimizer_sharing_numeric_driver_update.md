# FermiNet PBC Optimizer, Sharing, Numeric, And Driver Update

Date: 2026-05-25

## Summary

Completed a source-level iteration on the FermiNet PBC external-state
excited-state path after the first controlled driver trajectory.  The update
focuses on four gaps relative to the paper-style implementation path:
optimizer flexibility, shared-parameter projection, numerical scaling/guards,
and driver/runtime speed controls.

## Implemented

- Added lightweight external-state optimizer state and generalized optimizer
  stepping for SGD, Adam, LAMB, and direct KFAC.
- Added a direct external-state KFAC bridge using `kfac_jax.Optimizer` on the
  paper-tangent loss.  The KFAC step uses tuple-packed sample batches for
  internal JIT compatibility, registers external-state wavefunction outputs as
  normal predictive factors, reuses precomputed true penalty terms so local
  energy is not re-entered inside KFAC tracing, and copies donated parameter
  trees so shared leaves do not alias the same JAX buffer.
- Added EWMA state-energy and state-energy-std statistics for overlap-gradient
  scaling, so the overlap penalty does not depend only on a noisy instantaneous
  mini-batch estimate.
- Added shared-parameter projection by parameter-path substrings, including a
  `*` mode for validation.  High-level fixed-sample and sampler-integrated
  loops now project shared leaves before initial terms are evaluated.
- Split diagnostics into optimizer update norm, total parameter delta norm, and
  shared-parameter projection norm.
- Added update caps, gradient caps, candidate-check cadence, and finite
  candidate checks to the driver/runner/validation scripts.
- Extended driver checkpoints to carry optimizer state and EWMA running stats.
- Extended trajectory summaries with optimizer, EWMA, candidate-check, and
  shared-parameter diagnostics while preserving compatibility with old 0070
  summary JSON.

## Validation

Static compile passed for the touched source and scripts:

```text
python -m py_compile \
  src/solidnes/excited_states/__init__.py \
  src/solidnes/excited_states/ferminet_pbc_adapter.py \
  src/solidnes/excited_states/ferminet_pbc_driver.py \
  src/solidnes/excited_states/ferminet_pbc_training.py \
  scripts/backends/run_ferminet_pbc_excited_driver.py \
  scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py \
  scripts/validation/summarize_ferminet_pbc_driver_run.py \
  scripts/validation/check_ferminet_pbc_driver_smoke.py
```

Cheap-local-energy CPU smoke checks passed in
`solidnes-ferminet-jax0101-cuda12`:

```text
check_ferminet_pbc_penalty_opt_smoke.py --optimizer adam \
  --overlap-ewma-decay 0.5 --param-share-keys '*' \
  --candidate-check-period 2 --max-grad-l2-norm 1000 \
  --max-update-l2-norm 1e-3

check_ferminet_pbc_penalty_opt_smoke.py --optimizer lamb \
  --overlap-ewma-decay 0.5 --param-share-keys '*' \
  --candidate-check-period 2 --max-grad-l2-norm 1000 \
  --max-update-l2-norm 1e-3

check_ferminet_pbc_driver_smoke.py ... --optimizer adam \
  --overlap-ewma-decay 0.5 --param-share-keys '*' \
  --candidate-check-period 2 --max-grad-l2-norm 1000 \
  --max-update-l2-norm 1e-3

check_ferminet_pbc_penalty_opt_smoke.py --optimizer kfac \
  --overlap-ewma-decay 0.5 --param-share-keys layers/streams \
  --candidate-check-period 2 --max-grad-l2-norm 1000 \
  --max-update-l2-norm 1e-3 --kfac-damping 0.001 \
  --kfac-norm-constraint 0.001 --kfac-invert-every 1

run_ferminet_pbc_excited_driver.py ... --optimizer kfac \
  --local-energy-source cheap --iterations 1 --walkers 2 \
  --param-share-keys layers/streams --kfac-damping 0.001 \
  --kfac-norm-constraint 0.001 --kfac-invert-every 1
```

The KFAC fixed-sample 2-step smoke kept the expected local-energy call count
at `10`, proving the KFAC bridge did not re-enter local energy inside the
paper-tangent value-and-grad/curvature path.  The KFAC driver smoke wrote a
checkpoint roundtrip of `3702890` bytes, and a production-runner resume from
`driver_iter_000001.pkl` to iteration 2 completed successfully.

The current smoke-network parameter paths are:

```text
envelope/0/sigma
envelope/1/sigma
layers/streams/0/double/{b,w}
layers/streams/0/single/{b,w}
layers/streams/1/double/{b,w}
layers/streams/1/single/{b,w}
layers/streams/2/single/{b,w}
orbital/0/w
orbital/1/w
```

For the next non-smoke run, `layers/streams` is the conservative first
parameter-sharing key: it shares the hidden feature extractor while leaving the
state-specific orbital/envelope leaves independent.

The 0070 controlled-run trajectory summarizer remained backward-compatible:

```text
iterations: 12
objective_final: -13.3468618393
acceptance_mean: 0.809027781089
grad_norm_max: 9789.92871094
final_overlap_offdiag: 0.483194023371
```

## Notes

No new SLURM or GPU job was submitted for this source iteration.  The current
interactive node reports CPU as the available JAX backend in the FermiNet
environment, with CUDA plugin initialization warnings.

The next production step is a run-0071 plan using direct KFAC, EWMA overlap
scaling, `--param-share-keys layers/streams`, update caps, and a
candidate-check period greater than one.
