# 0101 DeepQMC Merge-Keys 4GPU Sweep

Purpose: compare DeepQMC-style independent-state parameter sharing choices after
the diagonal excited-state fast paths were validated in 0100.

Common setup:

- PsiFormer paper model with fused-QKV JAX attention.
- x64/fp64 runtime.
- `vmc_overlap`, 2 states, independent state parameter trees.
- Diagonal MCMC trace, diagonal local energy, and diagonal overlap-JVP enabled.
- batch4096, 4 GPUs, 200 iterations.
- `profile_step_times: true`.

Runs:

| run | merge keys | Slurm job | initial placement | purpose |
| --- | --- | --- | --- | --- |
| `merge_layers` | `[layers]` | `134801` | `amdgpu40g/gpu006`, completed | Current mainline candidate: shared transformer body, state-specific orbital head. |
| `merge_none` | `[]` | `134802` | `amdgpu40g/gpu004`, completed | Fully independent states, maximum flexibility and highest parameter count. |
| `merge_embed` | `[layers/embed]` | `134803` | `amdgpu40g`, completed | Light sharing of only the input embedding path. |

Compare after completion:

- Stable step time from `train_stats.csv`, excluding step 0 and warm-up step 1.
- `energy_matrix.npy` ordering and drift.
- `overlap_matrix.npy` and `overlap_penalty_matrix.npy` off-diagonal size.
- `pmove`, NaN/reset behavior, and whether KFAC remains stable through 200 steps.

Completed summary:

| run | tail20 step | tail20 energy mean | final energy | final EW variance | final pmove |
| --- | ---: | ---: | ---: | ---: | ---: |
| `merge_none` | 1.216 s | -72.637 | -73.035 | 0.221 | 0.750 |
| `merge_layers` | 1.290 s | -72.338 | -72.917 | 0.373 | 0.749 |
| `merge_embed` | 1.221 s | -72.822 | -73.048 | 0.168 | 0.756 |

`merge_embed` is the current best candidate from this 200-step sweep: it has
the lowest tail variance, slightly better tail energy, and essentially the same
speed as the fully independent run. `merge_layers` is slower and worse on the
tail metrics.

Result caveat: `merge_none` and `merge_layers` are missing final appended NPY
matrix files because zero-byte local build artifacts were removed while those
jobs were already running. Their `train_stats.csv` files are complete, but final
overlap/energy matrices need a clean rerun or a longer follow-up run. The
`merge_embed` appended-NPY summary is available under
`runs/merge_embed/results/validation/native_ferminet_excited_summary.*`.
