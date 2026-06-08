# 0108 continuation result summary

Both 0108 continuation jobs completed with Slurm exit code `0:0` and reached step `29999`. The analysis concatenates the original 0107 steps `0..9999` with the 0108 continuation steps `10000..29999`. Ground/excited energies are sorted per step from `energy_matrix.npy`; spin uses `spin_state_0` and `spin_state_1` from `train_stats.csv`.

## Tail means

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin state0 | Last1000 spin state1 | Last1000 s/step | Last1000 |S_offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.38956332 | -75.18281067 | 5.626026 +/- 1.579027 | 5.569321 +/- 1.574981 | 0.007474 | 0.009054 | 1.6251 | 0.007998 |
| Fused QKV | -75.38409287 | -75.18260977 | 5.482634 +/- 1.492863 | 5.558044 +/- 1.559539 | 0.006864 | 0.008084 | 1.6437 | 0.007634 |

## Final row

| Variant | Final ground (Ha) | Final excited (Ha) | Final gap (eV) | Final spin state0 | Final spin state1 | Final symmetric offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.39475039 | -75.18422242 | 5.728758 | 0.008472 | 0.010252 | 0.000000 |
| Fused QKV | -75.41793347 | -75.21460083 | 5.532963 | 0.007513 | 0.006660 | -0.000000 |

## Plot artifacts

- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after5000_window1000.png`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after10000_window1000.png`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after20000_window1000.png`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_gap_rolling_after20000_window1000.png`

## Notes

- `bare_energy_matrix.npy` is empty by design because the full S2 matrix observable is disabled; the plotted state energies come from `energy_matrix.npy`.
- The final single row is noisier than the last-window means, so the last1000 and last5000 values are the preferred read.
- Spin remains close to the requested singlet target for both methods; fused-QKV ends with the lower last1000 spin values, while FermiNet QKV has a slightly larger last1000 gap.
