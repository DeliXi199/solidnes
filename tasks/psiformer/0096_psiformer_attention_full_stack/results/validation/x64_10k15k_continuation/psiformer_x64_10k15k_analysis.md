# PsiFormer x64 10k->15k Continuation Analysis

Window analyzed: steps 10000 through 14999. Both jobs completed successfully.

## Summary

| Variant | Node | Frames | Mean gap (eV) | Last1000 gap (eV) | Final gap (eV) | EW mean tail1000 (Ha) | EW var tail1000 | Offdiag | Swaps | sec/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet Q/K/V on gpu002 | gpu002/amdgpu80g | 5000 | 6.875 +/- 6.101 | 5.603 +/- 2.477 | 5.820 | -75.326565 | 0.00282 | 0.00514 | 0 | 1.095 |
| fused-QKV on gpu005 | gpu005/amdgpu40g | 5000 | 5.748 +/- 3.357 | 5.391 +/- 2.925 | 3.777 | -75.325791 | 0.00239 | 0.00622 | 0 | 1.441 |

## 1000-Step Gap Bins

| Variant | Steps | Gap mean (eV) | Gap std (eV) | Ground mean (Ha) | Excited mean (Ha) |
| --- | --- | ---: | ---: | ---: | ---: |
| FermiNet Q/K/V on gpu002 | 10000-10999 | 7.848 | 6.390 | -75.377646 | -75.089252 |
| FermiNet Q/K/V on gpu002 | 11000-11999 | 6.482 | 5.424 | -75.386962 | -75.148769 |
| FermiNet Q/K/V on gpu002 | 12000-12999 | 4.976 | 2.004 | -75.401131 | -75.218263 |
| FermiNet Q/K/V on gpu002 | 13000-13999 | 9.468 | 9.634 | -75.371874 | -75.023920 |
| FermiNet Q/K/V on gpu002 | 14000-14999 | 5.603 | 2.477 | -75.396328 | -75.190433 |
| fused-QKV on gpu005 | 10000-10999 | 6.326 | 3.492 | -75.390532 | -75.158067 |
| fused-QKV on gpu005 | 11000-11999 | 5.798 | 3.130 | -75.391220 | -75.178150 |
| fused-QKV on gpu005 | 12000-12999 | 5.795 | 4.433 | -75.390092 | -75.177130 |
| fused-QKV on gpu005 | 13000-13999 | 5.431 | 2.370 | -75.397452 | -75.197873 |
| fused-QKV on gpu005 | 14000-14999 | 5.391 | 2.925 | -75.393429 | -75.195298 |

## Files

- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_states_gap.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_gap_smoothed_zoom3_10.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_training_stats.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_gap_1000step_bins.png`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_continuation_summary.csv`
- `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k15k_continuation/psiformer_x64_10k15k_continuation_timeseries.csv`
