# Spin Beta Sweep 0082-0085

```text
system: diamond C primitive cell, Gamma point
backend: native FermiNet PBC excited-state vmc_overlap
fixed settings: batch4096, KFAC, 1000 iterations, alpha=4.0, max_gap_std overlap scaling
changed setting: spin_penalty beta
baseline: 0078, same 1000-step no-spin run
jobs: 129309, 129310, 129311, 129312 on amdgpu80g/gpu002
```

| variant label | beta | status | final bare gap (eV) | tail100 bare gap mean (eV) | tail100 gap std (eV) | final S2 diag | tail100 S2 diag mean | final sym overlap01 |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| 0078 | none | finite baseline | 17.699 | 17.647 | 3.587 | n/a | n/a | n/a |
| 0082 | 0.02 | finite but unstable gap sign/noisy tail | -3.780 | 13.551 | 10.034 | [0.888, 0.597] | [0.506, 1.102] | 0.0077 |
| 0083 | 0.05 | numerical failure; NaN after about step 846 | nan | nan | nan | [nan, nan] | [nan, nan] | nan |
| 0084 | 0.10 | finite but gap too large | 25.235 | 31.783 | 29.609 | [1.004, 1.169] | [1.119, 0.278] | -0.0175 |
| 0085 | 0.20 | finite but gap and S2 too large/noisy | 78.437 | 34.550 | 100.742 | [7.025, 1.520] | [1.427, 1.036] | -0.0213 |

Conclusion: none of beta = 0.02, 0.05, 0.10, 0.20 is ready for a 10000-step
production run with the current learning-rate/KFAC settings. The best finite
low-beta point is 0.02, but its final gap is negative and the tail100 gap is
still much too noisy. Beta 0.05 produced NaNs, while beta 0.10 and 0.20 pushed
the bare gaps far above the expected diamond Gamma direct excitation scale.
