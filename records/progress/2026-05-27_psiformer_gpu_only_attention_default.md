# PsiFormer GPU-Only Attention Default

Date: 2026-05-27

This is a post-0095 source-policy update and does not allocate a new task
number.

## Change

The PsiFormer attention `auto` policy now resolves directly to `fused_qkv`.
The fallback for missing `cfg.network.psiformer_attention_implementation` is
also `auto`, so older PsiFormer configs without an explicit attention field
now enter the fused-QKV path by default.

The explicit `ferminet` option is still available for ablation/control runs.

## Rationale

Future PsiFormer calculations are GPU-only. Therefore CPU timing should not
control the default production path. Task 0094 showed exact output agreement
between upstream FermiNet attention and fused-QKV, with a positive GPU
forward-median speed result. Task 0095 showed that short full-training timing
is dominated outside the attention projection, not that upstream attention is
preferable for production.

## Validation

- `py_compile` passed for the changed PsiFormer attention module and related
  validation scripts.
- `git diff --check` passed.
- Build-only config check passed for
  `configs/experiment/diamond_c_psiformer_pbc_gamma_native_training_smoke_auto.yaml`.
- Direct install probe with `JAX_PLATFORMS=cpu` returned `fused_qkv`, confirming
  that `auto` is no longer backend-conditional.
