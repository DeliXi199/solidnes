# Task 0088 Recent Energy and Gap Summary

Spin diagnostics are intentionally excluded here. Statistics are computed from recent contiguous training windows, not from the final sample only.

## Recent Window Means

| window | steps | scalar E mean +/- block SE (Ha) | state0 mean (Ha) | state1 mean (Ha) | gap mean +/- block SE (eV) | gap median / 5-95% (eV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tail50 | 99950-99999 | -75.010937 +/- 0.010908 | -75.103948 | -74.825094 | 7.588 +/- 0.639 | 7.819 / [1.781, 13.018] |
| tail100 | 99900-99999 | -74.967494 +/- 0.016960 | -75.082836 | -74.737503 | 9.397 +/- 0.732 | 9.361 / [2.575, 15.103] |
| tail200 | 99800-99999 | -74.983151 +/- 0.013842 | -75.096070 | -74.758434 | 9.188 +/- 0.527 | 9.205 / [3.745, 14.876] |
| tail500 | 99500-99999 | -74.991765 +/- 0.018054 | -75.116989 | -74.743495 | 10.163 +/- 0.661 | 9.850 / [4.529, 16.838] |
| tail1000 | 99000-99999 | -74.903293 +/- 0.047000 | -75.035068 | -74.642503 | 10.682 +/- 0.669 | 10.183 / [4.181, 18.774] |
| tail5000 | 95000-99999 | -74.961617 +/- 0.024251 | -75.081659 | -74.723761 | 9.739 +/- 0.329 | 9.255 / [3.888, 17.653] |
| tail10000 | 90000-99999 | -74.970026 +/- 0.010257 | -75.086988 | -74.738254 | 9.490 +/- 0.199 | 9.077 / [3.771, 16.712] |

## Final Point for Reference

- Final scalar energy: `-75.037605 Ha`.
- Final EW mean: `-75.033005 Ha`.
- Final state energies: `[-75.096855, -74.919098] Ha`.
- Final gap: `4.837 eV`.

## Interpretation

- The final gap is lower than the recent-window average; it should not be used alone.
- The last 200 to 10000 samples give a gap centered near 9 eV, with several-eV fluctuation width.
- The scalar energy is stable near -75.03 to -75.04 Ha over the recent windows.
