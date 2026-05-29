# PsiFormer x64 10k 2x2 Method x Partition Comparison

All four jobs completed successfully. Each run uses x64/fp64, batch 4096, 4 GPUs, 64 CPUs, levmap128, pure-JAX attention, no spin penalty, and no S2 observable.

## Runtime

| Method | Partition | Node | Job | Train s/step | Slurm s/step | Slurm elapsed |
| --- | --- | --- | ---: | ---: | ---: | --- |
| FermiNet Q/K/V | 40G | gpu005 | 132575 | 1.232 | 1.240 | 03:26:38 |
| FermiNet Q/K/V | 80G | gpu002 | 132947 | 1.057 | 1.062 | 02:56:59 |
| fused-QKV | 40G | gpu005 | 132948 | 1.257 | 1.260 | 03:30:03 |
| fused-QKV | 80G | gpu002 | 132574 | 1.082 | 1.084 | 03:00:42 |

Runtime observations:

- Same node comparison: fused-QKV is 2.0% slower than FermiNet Q/K/V on 40G, and 2.3% slower on 80G by train runtime.
- Node effect: 40G is 16.5% slower than 80G for FermiNet Q/K/V, and 16.2% slower for fused-QKV.
- In this 2x2 set, node/partition effect is much larger than the QKV implementation effect.

## Result Summary

| Method | Partition | Final gap (eV) | Last1000 gap (eV) | Last500 gap (eV) | EW mean tail1000 (Ha) | EW var tail1000 | Swaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet Q/K/V | 40G | 4.046 | 8.349 +/- 6.888 | 6.628 +/- 3.350 | -75.260227 | 0.02427 | 14 |
| FermiNet Q/K/V | 80G | 3.112 | 7.333 +/- 4.740 | 6.028 +/- 2.285 | -75.283779 | 0.00641 | 9 |
| fused-QKV | 40G | 2.780 | 6.605 +/- 2.632 | 6.958 +/- 2.888 | -75.301177 | 0.00318 | 9 |
| fused-QKV | 80G | 4.461 | 6.881 +/- 3.122 | 6.419 +/- 2.391 | -75.295939 | 0.00439 | 9 |

Result observations:

- fused-QKV has smaller last1000 gap standard deviation on both nodes than FermiNet Q/K/V in these runs.
- State ordering swaps occur only in the first few iterations: FermiNet 40G last swap step 14, FermiNet 80G step 9, fused-QKV both step 9. No late root flipping is indicated.
- Final single-step gaps vary substantially and are less reliable than window averages; last1000 means put all four runs in the roughly 6.6-8.3 eV range.
- fused-QKV tail EW mean is more negative on both nodes, but these are independent fresh stochastic optimizations, so this should be treated as trajectory evidence rather than a deterministic accuracy proof.

## Files

- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_speed.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_gap_bars.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_gap_trajectories.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_state_energies.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_training_stats.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_summary.csv`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_timeseries.csv`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_1000step_bins.csv`
