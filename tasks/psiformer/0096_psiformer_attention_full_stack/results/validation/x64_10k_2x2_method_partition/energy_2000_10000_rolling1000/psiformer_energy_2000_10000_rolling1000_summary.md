# PsiFormer energy evolution, 1000-step rolling average

Source: `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_timeseries.csv`

Requested window: 2000-10000. The 10k runs contain steps 0-9999, so the actual plotted window is 2000-9999.

Rolling definition: trailing mean over the current step and previous 999 logged steps, computed on the full run before cropping to the display window.

| variant | node/job | ground rolling last (Ha) | excited rolling last (Ha) | gap rolling last (eV) | gap rolling mean ± std over displayed window (eV) |
|---|---:|---:|---:|---:|---:|
| FermiNet Q/K/V / 40G | gpu005 / 132575 | -75.373845 | -75.067027 | 8.349 | 7.739 ± 0.852 |
| FermiNet Q/K/V / 80G | gpu002 / 132947 | -75.376434 | -75.106956 | 7.333 | 6.922 ± 1.475 |
| fused-QKV / 40G | gpu005 / 132948 | -75.383242 | -75.140508 | 6.605 | 7.818 ± 1.155 |
| fused-QKV / 80G | gpu002 / 132574 | -75.381390 | -75.128512 | 6.881 | 8.160 ± 1.461 |
