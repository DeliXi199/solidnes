# 0110 damping 2e-3 continuation result summary

Both 0110 continuation jobs completed with Slurm exit code `0:0` and reached step `39999`.
The analysis uses the 0110 continuation steps `30000..39999`.
Ground/excited energies are sorted per step from the appended `energy_matrix.npy` stream; spin uses `spin_state_0` and `spin_state_1` from `train_stats.csv`.

## Tail means

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin | Last1000 spin state0 | Last1000 spin state1 | Last1000 s/step | Last1000 |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.39309419 | -75.18852078 | 5.566726 +/- 1.380652 | 5.599056 +/- 1.416943 | 0.006235 | 0.005364 | 0.007106 | 0.5742 | 0.007186 |
| Fused QKV | -75.39636183 | -75.18883065 | 5.647211 +/- 1.410595 | 5.611575 +/- 1.454483 | 0.006337 | 0.004768 | 0.007906 | 0.5780 | 0.007162 |

## Final row

| Variant | Final ground (Ha) | Final excited (Ha) | Final gap (eV) | Final spin state0 | Final spin state1 | Final symmetric offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.39791555 | -75.23327265 | 4.480162 | 0.005480 | 0.006345 | 0.026776 |
| Fused QKV | -75.38857819 | -75.19687249 | 5.216578 | 0.003977 | 0.008224 | 0.000000 |

## Plot artifacts

- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp2e3_energy_gap_spin_rolling_after30000_window1000.png`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp2e3_gap_rolling_after30000_window1000.png`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp2e3_energy_gap_spin_rolling_after35000_window1000.png`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp1e3_to_damp2e3_energy_gap_spin_rolling_after10000_window1000.png`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp1e3_to_damp2e3_gap_rolling_after10000_window1000.png`

## Notes

- `bare_energy_matrix.npy` is empty because the full S2 matrix observable is disabled; the plotted state energies come from the appended `energy_matrix.npy` stream.
- The final single row is noisier than the last-window means, so the last1000 and last5000 values are the preferred read.
