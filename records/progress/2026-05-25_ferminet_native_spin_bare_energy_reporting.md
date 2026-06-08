# FermiNet Native Spin Bare-Energy Reporting

Date: 2026-05-25, Asia/Shanghai

Current status note, 2026-06-01: this record describes the historical 0081
reporting smoke. The current native FermiNet/PsiFormer spin-penalty path is
DeepQMC-aligned and applies `beta * <S^2>` at the loss/JVP level rather than by
training on `H + beta S^2`; see
`records/progress/2026-06-01_spin_penalty_deepqmc_alignment.md`.

## Summary

At the time, the smoke aligned spin-penalized native FermiNet reporting with
the paper workflow:

```text
historical training loss used: H + beta S^2
reported physical gaps used: H
```

The then-current train loop wrote `bare_energy_matrix.npy` for spin-penalized
excited-state runs. It was derived from the training energy diagnostics and the
averaged `S^2` observable:

```text
state-specific path: E_bare_i = E_train_i - beta * diag(S^2)_i
matrix path:         H_bare_ij = H_train_ij - beta * S^2_ij
```

The native summary script from that pass reported:

```text
final_training_state_energy
final_training_gap_hartree
final_training_gap_ev
final_bare_state_energy
final_bare_gap_hartree
final_bare_gap_ev
```

`final_state_energy` used the bare energy when `bare_energy_matrix.npy` was
available.

## Validation

CPU/source checks:

```text
python -m compileall external/ferminet/ferminet/train.py \
  scripts/validation/check_ferminet_native_overlap_loss_alignment.py \
  scripts/validation/summarize_ferminet_native_excited_run.py

JAX_PLATFORMS=cpu python scripts/validation/check_ferminet_native_overlap_loss_alignment.py

JAX_PLATFORMS=cpu python scripts/backends/run_ferminet_train.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_bare_smoke.yaml \
  --build-only
```

GPU smoke:

```text
Run: 0081
Task root: tasks/excited_state_nesvmc/0081_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_bare_smoke
Job: 129308
State: COMPLETED
Exit code: 0:0
Elapsed: 00:02:00
Node: amdgpu80g/gpu002
GPUs: 4 x A100 80GB
```

Final smoke diagnostics:

```text
rows: 2
bare_energy_matrix_frames: 2
final_training_state_energy: [-22.508650, -22.803316]
final_bare_state_energy: [-22.741909, -23.123356]
final_training_gap_hartree: -0.294666
final_bare_gap_hartree: -0.381447
final_s2_diagonal: [2.332586, 3.200397]
final_s2_trace: 5.532983
```

The smoke is a reporting-path check only. The two-step energies are not a
physical excitation result.
