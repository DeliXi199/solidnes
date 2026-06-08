# 0107 Result Summary

Data source: jobs `136006` and `136007`, both completed with exit code `0:0`.

The analysis uses `energy_matrix.npy` for bare Hamiltonian state energies.
Because DeepQMC-style spin penalties are added at the loss level, the energy
matrix is not spin-penalized. The full S2 observable was disabled, so
`bare_energy_matrix.npy` is empty by design.

Rolling curves are computed on data with `step >= 2000` using a full 1000-step
window, so the plotted x-range starts at step 2999.

| Method | Ground E last-1000 mean (Ha) | Excited E last-1000 mean (Ha) | Gap last-1000 mean (eV) | Spin state 0 last-1000 | Spin state 1 last-1000 | pmove last-1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.35222136 | -75.14535968 | 5.628993 | 0.015620 | 0.016001 | 0.535115 |
| Fused QKV | -75.35223712 | -75.14132292 | 5.739268 | 0.015138 | 0.019332 | 0.535050 |

Notes:

- The ground-state energy is essentially identical between the two methods in
  the final 1000-step mean.
- The fused-QKV excited state is about `0.00404 Ha` higher than FermiNet-QKV in
  the final 1000-step mean, corresponding to a gap difference of about
  `0.110 eV`.
- Both spin states remain close to the target `S^2 = 0`; fused-QKV state 1 is
  slightly higher but still small.
- The symmetric overlap off-diagonal is zero on the final saved step for both
  methods; the final-1000 mean absolute off-diagonal overlap is about `0.0116`
  for FermiNet QKV and `0.0109` for fused QKV.

Generated plot:

```text
tasks/psiformer/0107_attention_qkv_spin0_4gpu_10000_rerun/analysis/0107_energy_gap_spin_rolling_after2000_window1000.png
```
