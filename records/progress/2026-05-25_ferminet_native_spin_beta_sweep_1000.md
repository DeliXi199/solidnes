# Native FermiNet Spin-Penalty Beta Sweep

Date: 2026-05-25

## Setup

Four 1000-step native FermiNet PBC excited-state `vmc_overlap` jobs were run for
diamond C at Gamma with the paper-aligned overlap settings fixed:

```text
batch_size: 4096
iterations: 1000
optimizer: kfac
overlap_penalty: 4.0
overlap_scale_by: max_gap_std
overlap_clip_width: 10.0
spin_penalty beta: 0.02, 0.05, 0.10, 0.20
observables_s2: true
bare_energy_matrix.npy: enabled
```

The no-spin 1000-step run `0078` is the same-length baseline.

The four beta variants are consolidated into one task bundle:

```text
tasks/excited_state_nesvmc/0082_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter1000/
```

Variant-specific artifacts live under `runs/beta002`, `runs/beta005`,
`runs/beta010`, and `runs/beta020`; aggregate comparison tables and plots live
under the parent `results/validation/` directory.

## Jobs

| variant label | beta | job | state | elapsed | node |
| --- | ---: | ---: | --- | --- | --- |
| 0082 | 0.02 | 129309 | COMPLETED | 00:04:55 | gpu002 |
| 0083 | 0.05 | 129310 | COMPLETED | 00:04:52 | gpu002 |
| 0084 | 0.10 | 129311 | COMPLETED | 00:04:56 | gpu002 |
| 0085 | 0.20 | 129312 | COMPLETED | 00:05:00 | gpu002 |

## Comparison

| variant label | beta | status | final bare gap (eV) | tail100 bare gap mean (eV) | tail100 gap std (eV) | final S2 diag | tail100 S2 diag mean |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 0078 | none | finite baseline | 17.699 | 17.647 | 3.587 | n/a | n/a |
| 0082 | 0.02 | finite but gap sign/noise unstable | -3.780 | 13.551 | 10.034 | [0.888, 0.597] | [0.506, 1.102] |
| 0083 | 0.05 | numerical failure; NaN after about step 846 | nan | nan | nan | [nan, nan] | [nan, nan] |
| 0084 | 0.10 | finite but gap too large | 25.235 | 31.783 | 29.609 | [1.004, 1.169] | [1.119, 0.278] |
| 0085 | 0.20 | finite but gap and S2 too large/noisy | 78.437 | 34.550 | 100.742 | [7.025, 1.520] | [1.427, 1.036] |

## Conclusion

The tested spin penalty values are not ready for a 10000-step production run
with the current learning-rate/KFAC settings. `beta=0.02` is the least bad
finite point, but its final bare gap is negative and its tail100 bare gap still
has a large standard deviation. `beta=0.05` produced NaNs, and `beta=0.10` and
`beta=0.20` produced finite trajectories with much too large bare gaps.

The next useful step is not a long run at these settings. It is a smaller,
stability-focused sweep around `beta <= 0.02`, likely paired with a reduced
KFAC norm constraint or learning rate, before committing to another 10000-step
run.
