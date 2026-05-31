# 0098 DeepQMC Alignment GPU Smoke

Purpose: validate the source-level DeepQMC alignment changes on real Slurm GPU
jobs before launching longer production calculations.

Runs:

- `psiformer`: PsiFormer paper model, two-state `vmc_overlap`,
  independent per-state params, `merge_keys: [layers]`, batch512, iter2.
- `ferminet`: native FermiNet small model, two-state `vmc_overlap`,
  independent per-state params, `merge_keys: [layers/streams]`, batch512,
  iter2.

Validation:

- Initial Slurm attempt:
  - `134778` (`0098-psi-deepqmc-smoke`) failed at the first KFAC update.
  - `134779` (`0098-fermi-deepqmc-smoke`) failed at the first KFAC update.
  - Root cause: selected merge-key leaves reused the same JAX buffer across
    independent state parameter trees, which conflicted with pmapped donation.
- Retry after making merged leaves value-equal but buffer-independent:
  - `134782` (`0098-fermi-deepqmc-r1`) completed on `amdgpu40g/gpu006`,
    elapsed `00:02:43`, `MaxRSS=4025332K`.
  - `134783` (`0098-psi-deepqmc-r1`) completed on `amdgpu40g/gpu006`,
    elapsed `00:04:43`, `MaxRSS=4866628K`.
- Output checks:
  - Both runs wrote 2 rows to `train_stats.csv`.
  - `energy_matrix.npy`, `overlap_matrix.npy`, `overlap_penalty_matrix.npy`,
    and `overlap_symmetric_matrix.npy` are present and finite for both runs.
