# Native FermiNet Beta 0 100000-Step Baseline Complete

Date: 2026-05-26, Asia/Shanghai

Task:
`tasks/excited_state_nesvmc/0088_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta0000_iter100000/`

Slurm job `129450` completed on `amdgpu80g/gpu002` with exit code `0:0` in
`03:46:24`. The run wrote all 100000 training rows plus the expected native
FermiNet matrix diagnostics.

Configuration:

- Native FermiNet PBC excited-state `vmc_overlap`.
- Carbon diamond primitive cell at Gamma.
- 2 states, batch4096, KFAC.
- Overlap alpha 4.0 with `max_gap_std` scaling.
- `spin_penalty=0.0`.
- `S^2` diagnostics enabled.

Main numbers:

- Final scalar energy: `-75.037605 Ha`.
- Final EW mean / variance: `-75.033005 Ha / 0.00333570 Ha^2`.
- Final state energies: `[-75.096855, -74.919098] Ha`.
- Final gap: `4.837 eV`.
- Tail200 gap median / 5-95%: `9.205 eV / [3.745, 14.876] eV`.
- Tail10000 gap median / 5-95%: `9.077 eV / [3.771, 16.712] eV`.
- Final `S^2` diagonal / trace: `[1.4585, 81.4612] / 82.9197`.
- Full-run `S^2` diagnostics contain 35 non-finite frames, or 140 matrix
  entries.
- Last 10000 finite frames contain 139 frames with `|S^2 trace| > 10` and
  34 frames with `|S^2 trace| > 50`; 10 additional trace frames are
  non-finite.

Interpretation:

The long beta=0 baseline completed cleanly as a scheduler job and provides a
usable no-spin-penalty energy trajectory. It does not solve the excited-state
spin problem. The final and tail `S^2` diagnostics show that spin remains
uncontrolled without a spin constraint or a different state-targeting method.

Artifacts:

- `results/validation/native_ferminet_excited_summary.md`
- `results/validation/native_ferminet_excited_summary.json`
- `results/validation/spin_beta0000_0088_analysis.md`
- `results/validation/spin_beta0000_0088_analysis.json`
- `results/validation/spin_beta0000_0088_timeseries.csv`
- `results/validation/spin_beta0000_0088_evolution_overview.png`
- `results/validation/spin_beta0000_0088_last10000_state_energy.png`
- `results/validation/spin_beta0000_0088_last10000_gap_s2.png`
