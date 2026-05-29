# PsiFormer energy evolution, 1000-step averages

Source: `tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_timeseries.csv`

Requested window: 2000-10000. The 10k runs contain steps 0-9999, so the actual plotted window is 2000-9999.

| variant | node/job | ground last bin (Ha) | excited last bin (Ha) | gap last bin (eV) | gap mean ± std (eV) |
|---|---:|---:|---:|---:|---:|
| FermiNet Q/K/V / 40G | gpu005 / 132575 | -75.373845 | -75.067027 | 8.349 | 7.714 ± 0.507 |
| FermiNet Q/K/V / 80G | gpu002 / 132947 | -75.376434 | -75.106956 | 7.333 | 6.591 ± 0.764 |
| fused-QKV / 40G | gpu005 / 132948 | -75.383242 | -75.140508 | 6.605 | 7.546 ± 0.920 |
| fused-QKV / 80G | gpu002 / 132574 | -75.381390 | -75.128512 | 6.881 | 7.849 ± 1.352 |
