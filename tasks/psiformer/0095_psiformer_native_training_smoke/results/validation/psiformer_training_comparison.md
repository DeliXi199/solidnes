# PsiFormer Native Training Comparison

Task root: `tasks/psiformer/0095_psiformer_native_training_smoke`

| Variant | Status | Attention | Batch | Iter | Rows | Elapsed s | s/iter | Final E | Mean pmove |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_smoke | completed | fused_qkv | 256 | 2 | 2 | 170.252 | 85.1261 | -26.8519 | 0.915039 |
| ferminet_b512 | completed | ferminet | 512 | 5 | 5 | 184.273 | 36.8545 | -25.9783 | 0.916074 |
| fused_qkv_b512 | completed | fused_qkv | 512 | 5 | 5 | 185.375 | 37.075 | -26.2452 | 0.917109 |
| fused_qkv_b1024 | completed | fused_qkv | 1024 | 5 | 5 | 209.231 | 41.8462 | -26.2749 | 0.914922 |

## Comparisons

- `b512_seconds_per_iteration_speedup`: `0.994053`
- `b512_elapsed_speedup`: `0.994053`
- `fused_b512_to_b1024_seconds_per_iteration_ratio`: `1.12869`
