# PsiFormer Native Training Smoke

Date: 2026-05-27

Task: `tasks/psiformer/0095_psiformer_native_training_smoke/`

## Summary

Task 0095 validated the native PsiFormer training path after task 0094 added
the configurable self-attention implementation. The task ran build/config
checks, a tiny `auto` GPU training smoke, a matched upstream-vs-fused-QKV
batch512 comparison, and a fused-QKV batch1024 timing probe on the `test`
partition.

All scheduled jobs completed successfully on `test/test001`:

| Variant | Job | Batch | Iter | Attention | Slurm State | Slurm Time | Runtime s/iter |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| `auto_smoke` | 131661 | 256 | 2 | `auto -> fused_qkv` | `COMPLETED` | 00:03:01 | 85.126 |
| `ferminet_b512` | 131664 | 512 | 5 | `ferminet` | `COMPLETED` | 00:03:16 | 36.855 |
| `fused_qkv_b512` | 131666 | 512 | 5 | `fused_qkv` | `COMPLETED` | 00:03:16 | 37.075 |
| `fused_qkv_b1024` | 131667 | 1024 | 5 | `fused_qkv` | `COMPLETED` | 00:03:41 | 41.846 |

## Result

The route is wired through the full native training stack. The `auto` policy
resolves to `fused_qkv` on GPU, and KFAC registration confirms the fused path
uses `qkv_w` instead of separate `q_w`, `k_w`, and `v_w` blocks.

The batch512 full-training comparison is effectively tied and slightly favors
upstream in this short run: `fused_qkv`/upstream speedup is `0.994x` by
seconds per iteration. This is different from the task 0094 forward-only GPU
benchmark, where fused-QKV had `1.051x` median speedup with exact output
agreement. The conclusion is that five-step native training is bottlenecked
outside the attention projection: KFAC, local-energy/FOLX, MCMC, and short-run
JIT/setup dominate the wall time.

The implementation decision after the GPU-only clarification is: keep `auto`
as the default, but resolve it directly to `fused_qkv`. The explicit
`ferminet` setting remains available for controls. This preserves tested
outputs and puts the launch-reduction optimization on the production path for
larger GPU runs.

## Artifacts

- Comparison table:
  `tasks/psiformer/0095_psiformer_native_training_smoke/results/validation/psiformer_training_comparison.md`
- Bottleneck notes:
  `tasks/psiformer/0095_psiformer_native_training_smoke/results/validation/psiformer_bottleneck_notes.md`
- Aggregated JSON:
  `tasks/psiformer/0095_psiformer_native_training_smoke/results/validation/psiformer_training_comparison.json`

## Next

Allocate task `0096` for a longer full-node PsiFormer run:
`tasks/psiformer/0096_psiformer_native_training_longer/`. Use
`configs/model/psiformer_pbc_small.yaml` with `attention=auto`, native
`vmc_overlap`, KFAC/FOLX, two states, and a batch of 2048 or 4096 depending
on the available GPU memory. The first longer run should be at least 1000
iterations so JIT/setup overhead no longer dominates speed measurements.
