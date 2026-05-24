# PBC-HF Pretrain Next Step Decision

Created: `2026-05-24`

## Benchmark

- Scheduled GPU job: `128113`
- Config: `batch_size=512`, `pretrain_iterations=100`, `pretrain_basis=ccpvdz`, `pretrain_target_chunk_size=8192`
- Loss: `2.14205 -> 0.0179985`
- Mean pmove: `0.844638`
- Steady GPU pretrain step: `0.058384 s`
- Steady PySCF target eval: `0.051142 s` (`87.6%` of step)
- Steady JAX update: `0.006495 s`

The pretraining path is numerically useful at this scale: loss decreases and
pmove remains in a normal range. The bottleneck is the host-side PySCF target
evaluation, not the JAX optimizer update.

## Structured Logging

- New runs now write `pretrain_stats.csv` with one row per pretrain step.
- CPU regression check wrote 100 rows plus header at
  `tasks/ferminet_pretraining/0046_ferminet_pbc_hf_pretrain_benchmark_batch512_iter100/results/checkpoints/pretrain_stats.csv`.
- The new pretrain summarizer supports both `pretrain_stats.csv` and legacy
  Slurm stderr log parsing.

## JAX PBC GTO MVP

- `sto-3g`, Gamma, s/p-shell diamond AO/MO comparison now runs on CPU.
- `image_cutoff=1`:
  - AO max abs: `6.30810389098e-03`
  - occupied MO max abs: `4.89748889023e-03`
- `image_cutoff=2`:
  - AO max abs: `3.27596351268e-07`
  - AO RMS abs: `4.75531479233e-08`
  - occupied MO max abs: `3.03450163858e-07`
  - occupied MO RMS abs: `4.41629742910e-08`

The image-sum formula, AO normalization, and PySCF spherical ordering are aligned
for the `sto-3g` Gamma MVP. The `image_cutoff=1` error is dominated by finite
image truncation.

## Decision

Proceed with the JAX-native target path, but do it incrementally:

1. Wire the `sto-3g`, Gamma, `image_cutoff=2` evaluator into a pretrain target
   validation path without replacing the production `ccpvdz` PySCF backend.
2. Add a timing benchmark for the JAX target evaluator on GPU using the project
   Slurm policy before expanding basis support.
3. After that, extend to `ccpvdz` by adding d shells, checking PySCF spherical
   harmonic ordering, and validating contracted basis values before using it as
   the default pretrain target.
