# FermiNet Native Method Cleanup And 10000-Step Run

Date: 2026-05-25

Scope:

- Keep spin penalty out of scope for this pass.
- Optimize and organize the native FermiNet PBC `vmc_overlap` path from the
  method and implementation side.
- Run one longer 10000-step trajectory after source validation.

Implementation changes:

- Added a method profile, `szabo_noe_2024_penalty`, in the SolidNES FermiNet
  adapter. It centralizes the paper-aligned defaults for overlap penalty
  strength, scale mode, clipping, energy-based ordering, EWM overlap scaling,
  and state-count KFAC norm scaling.
- Extended native FermiNet data/loss plumbing so overlap-gradient scale and
  state ordering can use EWM state-energy/std inputs instead of only the
  current noisy step.
- Split overlap diagnostics into raw psi-ratio overlap, symmetrized overlap,
  squared penalty overlap, scale matrices, ordering vectors, and EWM scale
  histories. The raw overlap file is intentionally retained for debugging but
  is not the orthogonality metric.
- Initialized fixed-shape overlap EWM arrays before KFAC optimizer
  construction. This avoids KFAC trace mismatches between the first step and
  later steps.
- Added the stable implementation note:
  `docs/02_theory/szabo_noe_2024_penalty_vmc_alignment.md`.

Validation:

```text
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
.venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_iter10000.yaml --build-only
```

Both checks passed. The build-only summary confirmed `states=2`,
`objective=vmc_overlap`, `method_profile=szabo_noe_2024_penalty`,
`overlap_use_ewm_scale=True`, effective `kfac_norm_constraint=0.002`, and
`iterations=10000`.

Run:

- Task root:
  `tasks/excited_state_nesvmc/0079_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000/`
- First submit job `129267` failed after step 0 because KFAC saw non-equivalent
  processed jaxprs when EWM fields changed from `None` to arrays.
- After fixed-shape EWM initialization, retry job `129272` completed on
  `amdgpu40g/gpu005` in `00:20:11` with exit `0:0`.

Final 10000-step summary:

```text
rows: 10000
steps: 0 -> 9999
final energy: -74.583840
final EW mean: -74.684850
final EW variance: 0.00702351
mean pmove: 0.549417
pmove range: [0.49023438, 0.9165527]
final state energies: [-74.792580, -74.169594]
final symmetric overlap matrix: [[1.0, 0.0159107], [0.0159107, 1.0]]
final overlap penalty matrix: [[1.0, 0.000253152], [0.000253152, 1.0]]
final overlap-gradient scale: [[5.0, 5.0], [5.0, 5.0]]
final state ordering: [0, 1]
final EWM energy: [-74.927254, -74.249290]
final EWM std: [6.065403, 7.452642]
```

Output summary:

```text
tasks/excited_state_nesvmc/0079_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000/results/validation/native_ferminet_excited_summary.md
tasks/excited_state_nesvmc/0079_ferminet_native_vmc_overlap_kfac_paper_aligned_iter10000/results/validation/native_ferminet_excited_summary.json
```
