# PsiFormer Full-Node 10k Attention Comparison

Task root: `tasks/psiformer/0096_psiformer_attention_full_stack`

Both formal full-node jobs completed on `amdgpu40g/gpu006` with four GPUs,
64 CPU cores, batch4096, 10000 native KFAC/FOLX `vmc_overlap` iterations,
no pretraining, `spin_penalty=0.0`, and `observables_s2=false`.

| Variant | Job | Attention | State | Slurm elapsed | Runtime s | s/iter | Rows | Final E | Tail100 E mean | Tail1000 E mean | Mean pmove |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| upstream | `131735` | `ferminet` | completed `0:0` | 01:26:37 | 5148.791 | 0.514879 | 10000 | -75.221344 | -75.192025 | -75.190272 | 0.547101 |
| fused-QKV | `131736` | `fused_qkv` | completed `0:0` | 01:26:50 | 5182.615 | 0.518262 | 10000 | -75.096440 | -75.054145 | -74.953024 | 0.546525 |

Timing result:

- `fused_qkv` / upstream runtime ratio: `1.006569`.
- Speedup computed as upstream seconds per iteration divided by fused-QKV seconds per iteration: `0.993473x`.
- In this full training workload, fused-QKV was about `0.657%` slower, so it did not improve end-to-end training speed.

Stability checks:

- Both jobs wrote all 10000 `train_stats.csv` rows, steps 0--9999.
- Scalar training stats are finite for both jobs.
- Final diagnostic arrays are finite for both jobs: `energy_matrix.npy`, `overlap_matrix.npy`, `overlap_symmetric_matrix.npy`, `overlap_penalty_matrix.npy`, `overlap_gradient_scale.npy`, `overlap_state_ordering.npy`, `overlap_scale_energy_ewm.npy`, and `overlap_scale_std_ewm.npy`.
- Final symmetric overlap off-diagonal was `0.061794` for upstream and `0.102972` for fused-QKV.
- Final overlap-penalty off-diagonal was `0.003819` for upstream and `0.010603` for fused-QKV.

Iteration plots:

- Full trajectory: `psiformer_fullnode_10k_iteration_comparison_full.png` and `psiformer_fullnode_10k_iteration_comparison_full.svg`.
- Steps 1000--9999: `psiformer_fullnode_10k_iteration_comparison_after1000.png` and `psiformer_fullnode_10k_iteration_comparison_after1000.svg`.
- Last 1000 steps: `psiformer_fullnode_10k_iteration_comparison_last1000.png` and `psiformer_fullnode_10k_iteration_comparison_last1000.svg`.

State-energy and gap diagnostics:

`energy_matrix.npy` contains one appended two-state vector per training step, so
the ground/excited curves below use all 10000 frames. Since spin penalty and S2
observables are disabled in these jobs, no bare-energy correction was applied.
Ground/excited are computed by instantaneous energy sorting, so the gap is
`excited - ground`.

| Variant | Final ground | Final excited | Final gap Ha | Final gap eV | Tail100 gap eV | Tail1000 gap eV | Final rolling100 gap eV | Final ordering | Ordering swaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| upstream | -75.320374 | -75.023132 | 0.297241 | 8.088345 | 5.376632 | 5.333406 | 6.758008 | `[0, 1]` | 8 |
| fused-QKV | -75.211212 | -75.039047 | 0.172165 | 4.684846 | 6.422431 | 7.932063 | 4.873413 | `[1, 0]` | 9131 |

Gap comparison:

- Final single-step gap: upstream is larger by `0.125076 Ha` (`3.403499 eV`).
- Last-1000 mean gap: fused-QKV is larger by `0.095499 Ha` (`2.598657 eV`).
- Final 100-step rolling gap: upstream is larger by `0.069258 Ha` (`1.884595 eV`).
- Because fused-QKV changes the energy ordering for most late steps, use tail or
  rolling statistics rather than a single final row when judging the gap trend.

State/gap artifacts:

- Full trajectory: `psiformer_fullnode_10k_state_energy_gap_comparison_full.png` and `psiformer_fullnode_10k_state_energy_gap_comparison_full.svg`.
- Steps 1000--9999: `psiformer_fullnode_10k_state_energy_gap_comparison_after1000.png` and `psiformer_fullnode_10k_state_energy_gap_comparison_after1000.svg`.
- Last 1000 steps: `psiformer_fullnode_10k_state_energy_gap_comparison_last1000.png` and `psiformer_fullnode_10k_state_energy_gap_comparison_last1000.svg`.
- Timeseries CSV: `psiformer_fullnode_10k_state_gap_timeseries.csv`.
- Summary CSV/JSON: `psiformer_fullnode_10k_state_gap_summary.csv`, `psiformer_fullnode_10k_state_gap_summary.json`.

Conclusion:

The attention mechanism is fully runnable through the paper-scale 10000-step
GPU training workflow. The fused-QKV implementation remains exact in the
forward benchmark and runs stably in training, but it does not provide an
end-to-end speed improvement in the current native KFAC/FOLX training path.
The bottleneck is still outside the attention projection, so further speed
work should target profiling the full training step or reducing KFAC/local
energy/MCMC overhead rather than adding more attention-only variants.
