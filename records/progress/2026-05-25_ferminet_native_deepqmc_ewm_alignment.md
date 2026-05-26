# FermiNet Native DeepQMC EWM Alignment

Date: 2026-05-25

Motivation:

- Recheck the earlier claim that SolidNES EWM scaling was an implementation
  deviation from the Szabo-Noe/DeepQMC method.
- Use the reference DeepQMC form first, before adding extra engineering
  variants.

Reference finding:

- DeepQMC keeps `alpha = 4.0` as a fixed scalar overlap penalty strength.
- The automatic part is not a search over `alpha`; it is the scaling of the
  overlap-gradient tangent.
- DeepQMC's scale functions consume `energy_ewm` and `std_ewm`.
- DeepQMC updates those EWM values with a finite buffer and weights
  `alpha_k * prod_{l<k}(1-alpha_l)`, where
  `alpha_t = max(1 - max_alpha, 1 / (2 + t / decay_alpha))`.

Changes:

- Replaced the previous simple recursive overlap-scale EWM update in
  `external/ferminet/ferminet/train.py` with the DeepQMC-style finite-buffer
  EWM update.
- Kept fixed-shape NaN EWM means before optimizer construction so KFAC traces
  remain shape-stable.
- Updated
  `docs/02_theory/szabo_noe_2024_penalty_vmc_alignment.md` to state that
  `overlap_penalty=4.0` is fixed and the automatic component is overlap-gradient
  scaling, not automatic alpha search.
- Added a regression check for the DeepQMC-style EWM weights in
  `scripts/validation/check_ferminet_native_overlap_loss_alignment.py`.

Validation:

```text
.venv/ferminet-jax0101-cuda12/bin/python -m compileall external/ferminet/ferminet/train.py scripts/validation/check_ferminet_native_overlap_loss_alignment.py
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
.venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000.yaml --build-only
```

All checks passed.
