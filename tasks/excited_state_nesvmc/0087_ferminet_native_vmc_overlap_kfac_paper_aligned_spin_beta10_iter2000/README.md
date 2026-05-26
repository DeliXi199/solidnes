# Task 0087: Native FermiNet Spin Penalty Beta 10, 2000 Iterations

## Purpose

Run a single intentionally large spin-penalty pressure test after task `0086`
showed that low beta values did not keep the diamond two-state calculation
spin-clean. This task sets `spin_penalty=10.0` while keeping the paper-aligned
native FermiNet PBC excited-state `vmc_overlap` settings from task `0086`.

## Configuration

- System: carbon diamond primitive cell at Gamma.
- Backend: native FermiNet PBC excited-state `vmc_overlap`.
- States: 2.
- Batch size: 4096.
- Iterations: 2000.
- Optimizer: KFAC.
- Overlap penalty alpha: 4.0.
- Overlap scaling: `max_gap_std`.
- Spin penalty beta: 10.0.
- Bare Hamiltonian energy output: enabled.
- `S^2` diagnostics: enabled.

## Files

- Experiment config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_beta10_iter2000.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_spin_beta10_iter2000.yaml`

## Interpretation Note

This beta is much larger than all values in task `0086`; it is a pressure test,
not a production recommendation. A useful outcome is either stable reduction of
both state `S^2` values toward zero or a clear failure mode showing that this
spin penalty scale destabilizes the current optimizer.

## Results

Slurm job `129431` completed on `amdgpu80g/gpu002` with exit `0:0` in
`00:07:46` and wrote all 2000 training rows plus the expected matrix
diagnostics. The run is finite, but it should be treated as a pressure-test
failure mode rather than a production candidate.

Key observations:

- Final bare gap: `8.458 eV`; final spin-penalized training gap: `2.793 eV`.
- Final `S^2` diagonal: `[0.0109, -0.0099]`; tail200 mean:
  `[-0.1463, 0.3148]`.
- Tail200 bare gap is unstable: median `4.520 eV`, but mean/std
  `-119.552 +/- 1368.586 eV` because step 1946 produced a
  `-18445.091 eV` bare-gap spike.
- Tail200 had 24 frames with absolute bare gap above `100 eV`, 2 above
  `500 eV`, and 2 above `1000 eV`.

Analysis outputs:

- `results/validation/spin_beta10_0087_analysis.md`
- `results/validation/spin_beta10_0087_analysis.json`
- `results/validation/spin_beta10_0087_evolution_overview.png`
- `results/validation/spin_beta10_0087_last1000_state_energy.png`
- `results/validation/spin_beta10_0087_vs_0086_tail200_robust.png`
