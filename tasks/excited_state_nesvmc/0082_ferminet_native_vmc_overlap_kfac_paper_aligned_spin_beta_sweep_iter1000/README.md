# Spin Beta Sweep 0082

This task bundle groups the four comparable 1000-step spin-penalty beta runs
for the paper-aligned native FermiNet PBC excited-state path.

```text
shared setup: diamond C primitive cell, Gamma point
backend: native FermiNet vmc_overlap
optimizer: KFAC
batch_size: 4096
iterations: 1000
overlap_penalty: 4.0
changed parameter: spin_penalty beta
```

## Layout

```text
runs/beta002/  beta = 0.02, Slurm job 129309
runs/beta005/  beta = 0.05, Slurm job 129310
runs/beta010/  beta = 0.10, Slurm job 129311
runs/beta020/  beta = 0.20, Slurm job 129312

results/validation/  aggregate comparison tables and plots
```

The subrun names preserve the original Slurm job labels, but this is one
numbered task bundle. Future sweeps and ablations should use this same pattern:
one task root, `runs/<variant>/` for comparable variants, and aggregate
comparison artifacts under the task root's `results/validation/`.

## Main Outputs

- `results/validation/spin_beta_sweep_0082_0085_comparison.md`
- `results/validation/spin_beta_sweep_0082_0085_evolution_after300_rolling50.png`
- `results/validation/spin_beta_sweep_0082_0085_training_vs_bare_gap.png`

## Conclusion

None of the tested beta values is ready for a 10000-step production run at the
current optimizer settings. `beta=0.02` was finite but still noisy, `beta=0.05`
went NaN, and `beta=0.10/0.20` gave overly large bare gaps.
