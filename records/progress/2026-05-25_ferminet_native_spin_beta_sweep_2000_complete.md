# FermiNet Native Spin Beta Sweep 2000 Complete

Date: 2026-05-25, Asia/Shanghai

## Task

Task `0086`:
`tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/`

The sweep tested `spin_penalty.beta` values
`0.000,0.001,0.002,0.005,0.008,0.010,0.012,0.015,0.018,0.020,0.025,0.030`
for the native FermiNet PBC two-state `vmc_overlap` path on carbon diamond
Gamma, batch4096, KFAC, 2000 iterations, overlap alpha 4.0, bare-energy output,
and `S^2` diagnostics.

## Slurm Outcome

Final jobs `129314` and `129327--129337` all completed on
`amdgpu80g/gpu002` with exit code `0:0`. Each variant wrote 2000 rows to
`results/checkpoints/train_stats.csv`.

The initial pending jobs `129315--129325` had been cancelled before running and
resubmitted because they inherited a 64GB queue memory cap. The final submitted
jobs used full-node exclusive allocations without the memory cap.

## Analysis Artifacts

- Aggregate comparison:
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/spin_beta_sweep_0086_comparison.md`
- Aggregate CSV:
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/spin_beta_sweep_0086_comparison.csv`
- Per-step series:
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/spin_beta_sweep_0086_series.csv`
- Evolution plots:
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/spin_beta_sweep_0086_evolution_after300_rolling50.png`
  and
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/spin_beta_sweep_0086_gap_vs_beta.png`
- Last-1000 physical state-energy plots:
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/last1000_state_energy/spin_beta_sweep_0086_last1000_state_energy_plots.md`
  and
  `tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000/results/validation/last1000_state_energy/spin_beta_sweep_0086_last1000_state_energy_grid.png`

## Result

No beta is production-ready at the current optimizer settings. Tail gap noise is
still too large for a 10000-step commitment.

The least bad continuation candidate is `beta=0.008`:

```text
final bare gap: 15.936 eV
tail200 bare gap: 16.463 +/- 9.490 eV
tail200 gap min/max: [-7.114, 90.638] eV
final S2 diag: [0.937, 1.108]
tail200 S2 diag mean: [1.016, 0.801]
final symmetric overlap01: -0.0045
```

Useful controls:

- `beta=0.002`: cleanest finite low-beta control, tail200 bare gap
  `9.251 +/- 5.836 eV`.
- `beta=0.018`: higher-beta control with cleaner final `S^2` diagonal but
  comparable gap noise, tail200 bare gap `13.930 +/- 10.128 eV`.

Rejected/suspect cases:

- `beta=0.005`, `0.010`, and `0.015` show gap collapse or near-collapse
  behavior.
- `beta=0.001` overshoots the gap and has very noisy `S^2`.
- `beta=0.012` and `beta=0.030` had transient non-finite `S^2`/bare-energy
  diagnostic frames.

## Next Step

Do not launch a 10000-step spin run from this screen. The next controlled
calculation should rerun around `beta=0.008` with reduced optimizer
aggressiveness and keep `beta=0.002`/`0.018` as controls, or introduce
excited-state pretraining before the next long spin-penalty calculation.
