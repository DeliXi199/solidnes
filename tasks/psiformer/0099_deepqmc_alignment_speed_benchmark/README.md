# 0099 DeepQMC Alignment Speed Benchmark

Purpose: compare diagonal independent-state fast paths against the same
DeepQMC-aligned method with those paths disabled.

Runs:

- `psiformer_diag_on`: PsiFormer, independent per-state params,
  `merge_keys: [layers]`, diagonal MCMC/local-energy/overlap-JVP enabled,
  batch512, iter6, step profiling enabled.
- `psiformer_diag_off`: same PsiFormer config with diagonal paths disabled.
- `ferminet_diag_on`: native FermiNet, independent per-state params,
  `merge_keys: [layers/streams]`, diagonal paths enabled, batch512, iter6.
- `ferminet_diag_off`: same native FermiNet config with diagonal paths disabled.

Speed is summarized from `train_stats.csv` excluding step 0, so compile and
first-step warmup do not dominate the comparison.
