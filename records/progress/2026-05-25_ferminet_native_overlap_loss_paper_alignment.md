# FermiNet Native Overlap-Loss Paper Alignment

Date: 2026-05-25

## Summary

The native FermiNet PBC excited-state path now carries the paper-aligned
Szabo-Noe/DeepQMC overlap-loss details directly inside the FermiNet
`vmc_overlap` objective:

- overlap penalty default `alpha = 4.0` for SolidNES `vmc_overlap` configs;
- overlap-gradient scaling by `max_gap_std`, with `min_scale = 0.001` and
  `max_scale = 5.0`;
- psi-ratio/overlap-ratio median-MAD clipping for the overlap custom-JVP
  tangent;
- median-centered local-energy clipping via FermiNet's existing
  `clip_median = true`, `center_at_clip = true` path;
- energy-based state ordering diagnostics and optional ordering of the
  lower-state-detached upper-triangle overlap tangent.

## Source Changes

- `external/ferminet/ferminet/loss.py`
  - added clipped-geometric squared overlap for the forward penalty;
  - added overlap-ratio clipping and masked centered overlap tangent inputs;
  - added overlap-gradient scale helpers for `none`, `energy_gap`,
    `energy_std`, and `max_gap_std`;
  - added optional state ordering and unpermutation so the upper-triangle
    tangent updates the higher-energy state after sorting.
- `external/ferminet/ferminet/train.py`
  - passes the new overlap settings into `make_energy_overlap_loss`;
  - writes `overlap_gradient_scale.npy` and
    `overlap_state_ordering.npy` next to the native excited-state energy and
    overlap matrices.
- `src/solidnes/backends/ferminet_adapter.py`
  - exposes the new overlap settings in build summaries;
  - defaults SolidNES native `vmc_overlap` to `alpha = 4.0`,
    `scale_by = max_gap_std`, and the paper clipping/scaling constants unless
    overridden.
- `scripts/validation/check_ferminet_native_overlap_loss_alignment.py`
  - checks clipped-geometric overlap, `max_gap_std` scaling, ratio clipping,
    energy ordering, lower-state-detached direction, and a synthetic
    custom-JVP gradient smoke.
- `scripts/validation/summarize_ferminet_native_excited_run.py`
  - reports overlap-gradient scale and state-ordering frames.

## Configs

- Added short GPU smoke config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_smoke.yaml`
  using batch4096, 5 iterations, native KFAC, `alpha = 4.0`,
  `scale_by = max_gap_std`, overlap clipping, median local-energy clipping,
  and `overlap_sort_states_by = energy`.
- Added longer ready-to-run config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned.yaml`
  using the same method settings and 1000 iterations. Its output path is
  reserved for run `0078` after the `0077` smoke.

## Validation

Local checks:

```text
python -m compileall external/ferminet/ferminet/loss.py external/ferminet/ferminet/train.py src/solidnes/backends/ferminet_adapter.py scripts/validation/summarize_ferminet_native_excited_run.py scripts/validation/check_ferminet_native_overlap_loss_alignment.py
JAX_PLATFORMS=cpu python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_smoke.yaml --build-only
git diff --check
```

All passed.

Scheduled GPU smoke:

```text
Run: 0077
Job: 129257
Node: amdgpu40g/gpu005
Resources: 4 x A100 40GB, 64 CPU cores, exclusive allocation
Elapsed: 00:02:04
Exit: 0:0
Rows: 5
Final energy: -22.351885
Final state energy: [-22.548399, -21.983301]
Final overlap matrix: [[1.0, 0.0314396], [0.0647455, 1.0]]
Final overlap-gradient scale: [[5.0, 5.0], [5.0, 5.0]]
Final state ordering: [0, 1]
```

Summary artifacts:

```text
tasks/excited_state_nesvmc/0077_ferminet_native_vmc_overlap_kfac_paper_aligned_smoke/results/validation/native_ferminet_excited_summary.md
tasks/excited_state_nesvmc/0077_ferminet_native_vmc_overlap_kfac_paper_aligned_smoke/results/validation/native_ferminet_excited_summary.json
```

## Notes

The native FermiNet path computes `max_gap_std` from the current batch energy
and variance available inside FermiNet's loss function. DeepQMC's reference
implementation uses running EWM energy/std in its data dict. Matching that
stateful EWM behavior exactly would require extending FermiNet optimizer/train
state, but the active native path now has the same scaling law, clipping,
penalty strength, KFAC settings, and lower-state detach semantics.
