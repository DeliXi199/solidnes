# Task 0088 Spin Beta 0 Analysis

```text
job_id: 129450
status: completed; 100000 training rows; exit code 0:0; elapsed 03:46:24 on gpu002
spin_penalty_beta: 0.0
objective: native FermiNet PBC vmc_overlap, 2 states, KFAC, batch4096, 100000 iterations
```

## Key Metrics

| metric | value |
| --- | ---: |
| final scalar energy | -75.037605 Ha |
| final EW mean / variance | -75.033005 Ha / 0.003336 Ha^2 |
| mean pmove / tail200 pmove | 0.5380 / 0.5361 |
| final state energies | [-75.096855, -74.919098] Ha |
| final gap | 4.837 eV |
| tail200 gap mean/std | 9.188 +/- 3.345 eV |
| tail200 gap median / 5-95% | 9.205 eV / [3.745, 14.876] eV |
| tail10000 gap median / 5-95% | 9.077 eV / [3.771, 16.712] eV |
| final S2 diagonal / trace | [1.4585, 81.4612] / 82.9197 |
| S2 non-finite diagnostics | 35 frames / 140 matrix entries |
| tail200 S2 diagonal mean | [0.0982, 1.7654] |
| tail200 abs S2 trace > 10 / > 50 | 6 / 2 frames |
| tail10000 abs S2 trace > 10 / > 50 | 139 / 34 frames |
| final symmetric overlap01 | -0 |
| tail200 abs symmetric overlap01 mean/std | 0.00514537 +/- 0.00823076 |

## Interpretation

- The Slurm job completed normally and wrote all expected 100000 training rows plus matrix diagnostics.
- Energy is the best-behaved part of the run: the final two-state energies are -75.096855 Ha and -74.919098 Ha, with final gap 4.837 eV.
- Spin is not controlled at beta=0. The last snapshot jumps to S2 diagonal [1.4585, 81.4612], and the last observable trace is 82.9197.
- The tail is not just a single-line artifact: in the last 10000 frames, 139 finite frames have |S2 trace| > 10, 34 finite frames have |S2 trace| > 50, and 10 additional trace frames are non-finite.
- Across the full trajectory, S2 diagnostics have 35 non-finite frames, involving 140 matrix entries. Energy and overlap arrays stayed finite.
- Since beta=0 has no spin penalty, `bare_energy_matrix.npy` is absent and the training/physical state energy is `energy_matrix.npy`.

## Figures

- `spin_beta0000_0088_evolution_overview.png` / `.svg`: full-run scalar energy, state energies, gap, S2, overlap, variance, and pmove.
- `spin_beta0000_0088_last10000_state_energy.png` / `.svg`: last-10000 state energies.
- `spin_beta0000_0088_last10000_gap_s2.png` / `.svg`: last-10000 gap and S2 diagnostics.
- `spin_beta0000_0088_timeseries.csv`: flattened time series used for plotting.
