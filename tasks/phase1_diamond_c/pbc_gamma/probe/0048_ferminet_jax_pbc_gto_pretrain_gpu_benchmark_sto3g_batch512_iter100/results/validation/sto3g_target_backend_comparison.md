# STO-3G PBC-HF Target Backend Comparison

Created: `2026-05-24`

This compares the same FermiNet PBC pretraining setup with two target
evaluators:

- `0048`: `jax_pbc_gto`, `sto-3g`, Gamma, `image_cutoff=2`
- `0049`: `pyscf_pbc`, `sto-3g`, Gamma, host-side PySCF target

Both runs used `batch_size=512`, `pretrain_iterations=100`, one scheduled A100
80GB GPU, and warmup exclusion of the first 10 pretrain rows for steady-state
timing.

## Result

| Metric | 0048 JAX PBC GTO | 0049 PySCF PBC | Ratio PySCF/JAX |
| --- | ---: | ---: | ---: |
| Mean step seconds | 0.008500 | 0.045811 | 5.39x |
| Median step seconds | 0.008494 | 0.045846 | 5.40x |
| P90 step seconds | 0.009039 | 0.046647 | 5.16x |
| Mean target eval seconds | 0.002028 | 0.038902 | 19.19x |
| Mean JAX update seconds | 0.006262 | 0.006296 | 1.01x |
| Target transfer seconds | 0.000000 | ~0.0004-0.0006 | n/a |

The loss curves are effectively identical:

- JAX target: `2.1389451045 -> 0.0187578242`
- PySCF target: `2.1389451460 -> 0.0187578250`
- Mean pmove in both runs: `0.841660`

## Interpretation

For the validated `sto-3g` Gamma MVP, moving target evaluation into JAX reduces
target-eval time by about `19x` and total steady pretrain step time by about
`5.4x`. The optimizer/update portion is unchanged, as expected.

The benchmark does not yet imply the same speedup for `ccpvdz`: the current JAX
target has only been validated for `sto-3g` s/p shells. The next correctness
work is to validate d shells, PySCF spherical ordering, and cc-pVDZ contraction
before enabling `jax_pbc_gto` for production ccpvdz pretraining.
