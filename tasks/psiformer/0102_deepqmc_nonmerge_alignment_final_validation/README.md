# 0102 DeepQMC Non-Merge Alignment Final Validation

This task validates the DeepQMC-aligned excited-state path after freezing
`merge_keys` work. The current mainline parameter sharing remains fixed at
`independent_state_merge_keys: [layers]`; this task does not sweep or optimize
merge-key choices.

## Code Scope

- Added opt-in component profiling:
  - `local_energy_seconds`
  - `overlap_forward_seconds`
  - `kfac_update_seconds`
- Kept component profiling disabled by default in mainline production configs.
- Added adapter/runtime metadata for diagonal path and profiling switches.
- Added validation that KFAC overlap-JVP registration preserves `(batch, states)`
  rather than flattening state into the sample batch.
- Reused the DeepQMC-aligned diagonal fast paths:
  - diagonal MCMC trace
  - diagonal local energy
  - diagonal overlap JVP

## Slurm Runs

| Run | Job | GPUs | State | Stable step avg |
| --- | --- | ---: | --- | ---: |
| smoke_diag_on | 134805 | 1 | COMPLETED 0:0 | 0.3055 s |
| diag_on | 134806 | 4 | COMPLETED 0:0 | 1.2606 s |
| diag_off | 134807 | 4 | COMPLETED 0:0 | 2.1733 s |

Stable averages exclude step 0 and step 1.

## 4GPU Stable Timing Breakdown

| Path | step | MCMC | optimizer/KFAC | local energy probe | overlap probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| diag_on | 1.2606 s | 0.2441 s | 1.0165 s | 0.6183 s | 0.0254 s |
| diag_off | 2.1733 s | 0.5020 s | 1.6713 s | 1.1772 s | 0.0260 s |

The diagonal path is 1.72x faster than the full-matrix control on the 4096
sample, 4GPU benchmark, or about a 42% step-time reduction.

## Output Checks

- `train_stats.csv` includes the new profiling columns for all runs.
- Native excited-state summaries were generated for both 4GPU runs.
- Final overlap and energy matrices are finite.
