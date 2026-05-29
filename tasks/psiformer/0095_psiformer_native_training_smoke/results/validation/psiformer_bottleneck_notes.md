# PsiFormer Training Bottleneck Notes

Task root: `tasks/psiformer/0095_psiformer_native_training_smoke`

## What Ran

All runs used native FermiNet `vmc_overlap`, KFAC, FOLX, two states, the small
PsiFormer PBC model, and the carbon-diamond Gamma test configuration.

| Variant | Job | Batch | Iter | Attention | Elapsed s | s/iter | Slurm elapsed |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `auto_smoke` | 131661 | 256 | 2 | `auto -> fused_qkv` | 170.252 | 85.126 | 00:03:01 |
| `ferminet_b512` | 131664 | 512 | 5 | `ferminet` | 184.273 | 36.855 | 00:03:16 |
| `fused_qkv_b512` | 131666 | 512 | 5 | `fused_qkv` | 185.375 | 37.075 | 00:03:16 |
| `fused_qkv_b1024` | 131667 | 1024 | 5 | `fused_qkv` | 209.231 | 41.846 | 00:03:41 |

## Interpretation

The native training path is now validated on GPU for `auto`, upstream
FermiNet attention, and fused-QKV attention. The `auto` policy resolves to
`fused_qkv` on GPU as intended.

The b512 training-path comparison does not show a full-run speedup:
`ferminet_b512` is 36.855 s/iter and `fused_qkv_b512` is 37.075 s/iter
(`0.994x` speedup, effectively tied and slightly slower). This does not
contradict the task 0094 forward-only benchmark, where fused-QKV improved the
median forward time by `1.051x` at 256 walkers with exact output agreement.
In the full training loop, KFAC graph setup/curvature work, local-energy/FOLX
evaluation, MCMC, checkpoint/stat writing, and short-run JIT amortization
dominate enough that the attention projection launch reduction is not visible
at five iterations.

The fused-QKV path still reduces the attention parameter-registration shape:
upstream registers separate `q_w`, `k_w`, and `v_w` blocks per layer, while
fused-QKV registers one `qkv_w` block per layer. This is the safe optimization
to keep because it preserves the represented function and exact tested outputs.
After the GPU-only production clarification, `auto` now resolves directly to
`fused_qkv`; use the explicit `ferminet` setting only for controls.

## Next Experiment

Use task `0096` for a longer full-node PsiFormer run once a non-test GPU node
is available. Recommended first formal run:

- model: `configs/model/psiformer_pbc_small.yaml` (`attention=auto`)
- train: paper-aligned native `vmc_overlap`, KFAC/FOLX, two states
- batch: start at 2048 or 4096 depending on allocated GPU memory
- iterations: at least 1000 for stability and speed amortization
- output: `tasks/psiformer/0096_psiformer_native_training_longer/`

The 0095 data is too short to judge energy quality. Its purpose is complete:
prove the route runs, prove the attention implementation resolves correctly,
and quantify that short full-training timing is bottlenecked outside the
attention projection itself.
