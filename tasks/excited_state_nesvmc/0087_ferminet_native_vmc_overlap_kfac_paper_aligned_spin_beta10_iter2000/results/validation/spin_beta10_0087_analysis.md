# Task 0087 Spin Beta 10 Analysis

```text
job_id: 129431
status: completed; 2000 training rows; no non-finite arrays detected
spin_penalty_beta: 10.0
objective: native FermiNet PBC vmc_overlap, 2 states, KFAC, batch4096
```

## Key Metrics

| metric | value |
| --- | ---: |
| final scalar energy | -74.674130 Ha |
| final EW mean / variance | -74.728580 Ha / 0.203859 Ha^2 |
| mean pmove / tail200 pmove | 0.5982 / 0.5367 |
| final bare state energies | [-74.818260, -74.507423] Ha |
| final training state energies | [-74.709381, -74.606743] Ha |
| final bare gap | 8.458 eV |
| final training gap | 2.793 eV |
| tail200 bare gap mean/std | -119.552 +/- 1368.586 eV |
| tail200 bare gap median / 5-95% | 4.520 eV / [-109.208, 97.376] eV |
| tail200 worst bare gap | step 1946: -18445.091 eV |
| tail200 abs gap > 100 / 500 / 1000 eV | 24 / 2 / 2 frames |
| tail1000 bare gap median | 4.842 eV |
| tail200 negative bare gap fraction | 0.400 |
| final S2 diagonal / trace | [0.0109, -0.0099] / 0.0010 |
| tail200 S2 diagonal mean | [-0.1463, 0.3148] |
| tail200 abs symmetric overlap01 | 0.00436 +/- 0.00727 |
| final symmetric overlap01 | -0.01624 |

## Interpretation

- The beta=10 pressure test completed cleanly and wrote all expected matrix diagnostics; no non-finite values were found in the training energy, bare-energy, S2, or symmetric-overlap arrays.
- The large spin penalty does suppress the final spin diagnostic: final S2 diagonal is [0.0109, -0.0099]. Tail200 means are still not exactly zero ([-0.1463, 0.3148]) but are far below the noisy low-beta sweep scale.
- The physical bare gap is unstable. Tail200 median is 4.520 eV, but the mean/std are -119.552 +/- 1368.586 eV because step 1946 produced a -18445.091 eV bare-gap spike.
- The final snapshot alone is misleading: final bare gap is 8.458 eV and final training gap is 2.793 eV, but the tail contains repeated large physical-energy spikes. Use the time series, not just the final row, for this task.
- Compared with task 0086, beta=10 is useful as a pressure-test failure mode: it improves spin cleanliness and overlap suppression, but it does not produce a stable physical excitation gap under the current KFAC settings.

## Figures

- `spin_beta10_0087_evolution_overview.png` / `.svg`: full-run scalar energy, state energies, gap, S2, overlap, and pmove.
- `spin_beta10_0087_last1000_state_energy.png` / `.svg`: last-1000 two-state physical and training energies.
- `spin_beta10_0087_vs_0086_sweep.png` / `.svg`: mean/std comparison against task 0086; shows how strongly the 0087 gap outlier dominates standard statistics.
- `spin_beta10_0087_vs_0086_tail200_robust.png` / `.svg`: median and 5-95% robust comparison against task 0086.
- `spin_beta10_0087_timeseries.csv`: flattened time series used for plotting.
- `spin_beta10_0087_vs_0086_tail200_robust.csv`: robust comparison table.
