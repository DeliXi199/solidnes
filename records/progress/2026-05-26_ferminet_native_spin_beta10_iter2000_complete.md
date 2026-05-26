# FermiNet Native Spin Beta 10 Iter2000 Complete

Date: 2026-05-26

Task `0087` completed the deliberately large `spin_penalty=10.0` pressure test:

```text
task: tasks/excited_state_nesvmc/0087_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta10_iter2000
job_id: 129431
node: amdgpu80g/gpu002
slurm_state: COMPLETED
exit_code: 0:0
elapsed: 00:07:46
rows: 2000
status: completed; no non-finite arrays in training energy, bare energy, S2, or symmetric-overlap diagnostics
```

The run is finite but not production-ready. The large spin penalty suppresses
the final spin diagnostic, but the physical bare Hamiltonian gap is unstable:

- final bare gap: `8.458 eV`
- final spin-penalized training gap: `2.793 eV`
- final `S^2` diagonal: `[0.0109, -0.0099]`
- tail200 `S^2` diagonal mean: `[-0.1463, 0.3148]`
- tail200 bare-gap median: `4.520 eV`
- tail200 bare-gap mean/std: `-119.552 +/- 1368.586 eV`
- worst tail200 bare-gap spike: step `1946`, `-18445.091 eV`
- tail200 frames with absolute bare gap above `100/500/1000 eV`: `24/2/2`

Analysis and plots were written under the task validation directory:

- `results/validation/spin_beta10_0087_analysis.md`
- `results/validation/spin_beta10_0087_analysis.json`
- `results/validation/spin_beta10_0087_timeseries.csv`
- `results/validation/spin_beta10_0087_evolution_overview.png`
- `results/validation/spin_beta10_0087_last1000_state_energy.png`
- `results/validation/spin_beta10_0087_vs_0086_sweep.png`
- `results/validation/spin_beta10_0087_vs_0086_tail200_robust.png`

Conclusion: beta=10 is useful as a pressure-test failure mode. It improves
final spin cleanliness and overlap suppression, but it does not stabilize the
physical excitation gap under the current KFAC settings. The next controlled
spin-penalty run should reduce optimizer aggressiveness around `beta=0.008`,
or add excited-state pretraining before another long run.
