# Spin Beta Sweep 0086

```text
system: diamond C primitive cell, Gamma point
backend: native FermiNet PBC excited-state vmc_overlap
fixed settings: batch4096, KFAC, 2000 iterations, alpha=4.0, max_gap_std overlap scaling
changed setting: spin_penalty beta
jobs: 129314 and 129327--129337 on amdgpu80g/gpu002
```

| variant | beta | status | final bare gap (eV) | tail200 bare gap mean (eV) | tail200 gap std (eV) | tail200 gap min/max (eV) | final S2 diag | tail200 S2 diag mean | final sym overlap01 |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| beta0000 | 0.000 | finite but noisy | 14.032 | 20.263 | 19.741 | [-2.322, 116.869] | [0.915, -3.537] | [0.433, 3.484] | 0.0432 |
| beta0001 | 0.001 | finite but noisy | 24.591 | 29.163 | 26.854 | [-80.542, 111.979] | [1.583, 2.145] | [7.946, 1.256] | -0.0000 |
| beta0002 | 0.002 | finite | 14.270 | 9.251 | 5.836 | [-2.668, 36.194] | [1.053, 1.381] | [1.089, 1.321] | 0.0081 |
| beta0005 | 0.005 | finite but sign-unstable | 5.107 | 6.142 | 8.965 | [-73.119, 81.033] | [1.154, 4.515] | [1.051, 1.127] | 0.0310 |
| beta0008 | 0.008 | finite but noisy | 15.936 | 16.463 | 9.490 | [-7.114, 90.638] | [0.937, 1.108] | [1.016, 0.801] | -0.0045 |
| beta0010 | 0.010 | finite but noisy | 4.075 | 14.321 | 14.071 | [-5.791, 82.955] | [1.365, -0.356] | [1.062, 0.623] | 0.0094 |
| beta0012 | 0.012 | non-finite | 12.439 | 12.851 | 15.964 | [-6.810, 126.564] | [0.892, 0.794] | [1.862, 0.597] | -0.0127 |
| beta0015 | 0.015 | finite but noisy | 1.722 | 10.898 | 10.890 | [-70.780, 62.564] | [0.777, 2.603] | [1.984, 1.604] | 0.0236 |
| beta0018 | 0.018 | finite but noisy | 14.500 | 13.930 | 10.128 | [-9.557, 78.609] | [0.202, 0.037] | [0.749, 0.550] | -0.0101 |
| beta0020 | 0.020 | finite but noisy | 8.447 | 16.411 | 15.511 | [-58.648, 78.187] | [0.390, 0.683] | [1.346, 0.346] | -0.0089 |
| beta0025 | 0.025 | finite but noisy | 13.972 | 11.324 | 15.131 | [-155.384, 87.550] | [1.772, 1.075] | [0.921, 0.710] | -0.0210 |
| beta0030 | 0.030 | non-finite | 12.135 | 30.506 | 17.001 | [-45.985, 69.397] | [1.064, 0.832] | [0.748, 1.096] | 0.0242 |

## Interpretation

All 12 jobs completed and wrote 2000 training rows. No Slurm job failed and no log traceback was found, but `beta=0.012` and `beta=0.030` had transient non-finite `S^2`/bare-energy diagnostic frames, so they should not be treated as clean candidates.

No beta is production-ready at the current optimizer settings; the tail gap noise is still too large for a 10000-step commitment. If one continuation point must be chosen, `beta=0.008` is the least bad candidate: final bare gap `15.936 eV`, tail200 `16.463 +/- 9.490 eV`, small final symmetric overlap `-0.0045`, and tail200 `S^2` diagonal mean `[1.016, 0.801]`. `beta=0.018` has a cleaner final `S^2` diagonal but comparable gap noise, while `beta=0.002` is the most finite/stable low-beta control but gives a lower tail200 gap (`9.251 +/- 5.836 eV`).

Recommended next controlled reruns: do not launch a 10000-step spin run from this screen. Rerun `beta=0.008` with reduced optimizer aggressiveness and keep `beta=0.002`/`0.018` as controls, or move first to excited-state pretraining before the next long spin-penalty run.
