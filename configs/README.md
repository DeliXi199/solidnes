# Configs

Experiment settings live here.

Each config should be specific enough that a run can be reproduced without
guessing hidden parameters from a script.

## Layout

SolidNES uses role-split configs, inspired by reproducible VMC research
workflows.

```text
configs/
  experiment/   # top-level experiment configs that reference the others
  system/       # material, cell, electrons, boundary conditions
  model/        # ansatz and backend choices
  sampler/      # Monte Carlo sampler settings
  train/        # optimizer, steps, checkpoints, diagnostics
```

The experiment config is the entry point. It should reference one file from
each other config category.

## Current Phase 1 Entry Points

- `experiment/diamond_c_deepsolid_ground_smoke.yaml`
- `experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml`
- `experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot.yaml`
- `experiment/diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000.yaml`

## Current Backend Choice

The first concrete ground-state backend target is DeepSolid:

```text
experiment/diamond_c_deepsolid_ground_smoke.yaml
experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml
```

Current active work is carbon-diamond focused. New experiment configs should
write outputs under numbered `tasks/.../NNNN_slug/` bundles, not top-level
`results/`, `outputs/`, or `logs/`.
