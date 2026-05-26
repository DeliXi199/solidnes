# 0079 FermiNet Native Paper-Aligned 10000-Step Run

Date: 2026-05-25

Purpose: run the cleaned native FermiNet PBC two-state penalty-VMC method after
method-profile consolidation, EWM overlap-gradient scaling, state-count KFAC
norm scaling, and overlap diagnostic separation.

Configuration:

```text
configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000.yaml
```

Expected diagnostics:

```text
train_stats.csv
energy_matrix.npy
overlap_matrix.npy
overlap_symmetric_matrix.npy
overlap_penalty_matrix.npy
overlap_gradient_scale.npy
overlap_state_ordering.npy
overlap_scale_energy_ewm.npy
overlap_scale_std_ewm.npy
```

Result:

```text
Slurm job: 129272
Status: completed, exit 0:0
Node: amdgpu40g/gpu005
Runtime: 00:20:11
Rows: 10000
Final energy: -74.583840
Final EW mean: -74.684850
Final EW variance: 0.00702351
Mean pmove: 0.549417
Final state energies: [-74.792580, -74.169594]
Final symmetric overlap matrix: [[1.0, 0.0159107], [0.0159107, 1.0]]
Final overlap penalty matrix: [[1.0, 0.000253152], [0.000253152, 1.0]]
```

Notes:

- First submit job `129267` failed after step 0 because the KFAC optimizer saw
  non-equivalent traces when overlap EWM fields changed from `None` to arrays.
  The source was fixed by initializing fixed-shape NaN EWM arrays before KFAC
  optimizer construction, then retrying from a clean 0079 output directory.
- `overlap_matrix.npy` remains the raw non-symmetric psi-ratio diagnostic.
  Orthogonality should be read from `overlap_symmetric_matrix.npy` and
  `overlap_penalty_matrix.npy`.
- Iteration plots and derived scalar series:
  - `results/validation/native_ferminet_excited_iterations_full.png`
  - `results/validation/native_ferminet_excited_iterations_tail_after1000.png`
  - `results/validation/native_ferminet_excited_iteration_series.csv`
