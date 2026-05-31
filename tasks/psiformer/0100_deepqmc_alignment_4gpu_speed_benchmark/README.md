# 0100 DeepQMC Alignment 4GPU Speed Benchmark

Purpose: measure the production-like speed impact of the diagonal
independent-state fast paths using the 0096 PsiFormer paper-scale stack.

Common setup:

- PsiFormer paper model with fused-QKV JAX attention.
- x64/fp64 runtime.
- `vmc_overlap`, 2 states, independent state parameters, `merge_keys: [layers]`.
- batch4096, 4 GPUs, 8 iterations.
- `profile_step_times: true`.

Comparison:

- `diag_on`: diagonal MCMC trace, diagonal local energy, and diagonal overlap-JVP
  enabled.
- `diag_off`: same method and parameter structure with those diagonal paths
  disabled.

Speed should be summarized from `train_stats.csv` excluding step 0.

Completed runs:

| run | slurm job | partition | state | elapsed | stable step range |
| --- | --- | --- | --- | --- | --- |
| `diag_off` | `134798` | `amdgpu40g` | `COMPLETED` | `00:12:13` | steps 2-7 |
| `diag_on` | `134799` | `amdgpu40g` | `COMPLETED` | `00:08:15` | steps 2-7 |

Stable timing summary, batch4096 on 4 GPUs:

| metric | `diag_off` | `diag_on` | speedup | reduction |
| --- | ---: | ---: | ---: | ---: |
| total step | 2.165 s | 1.268 s | 1.71x | 41.4% |
| MCMC | 0.498 s | 0.243 s | 2.05x | 51.3% |
| optimizer/KFAC | 1.667 s | 1.025 s | 1.63x | 38.5% |
| post-step logging | 0.0176 s | 0.0180 s | 0.98x | -2.2% |

The diagonal path outputs were finite for energy, overlap, overlap penalty, and
symmetrized overlap matrices in both runs.
