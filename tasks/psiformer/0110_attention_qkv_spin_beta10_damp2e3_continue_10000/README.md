# 0110 Attention QKV Spin Beta-10 Damping 2e-3 Continuation

This task branches from the completed 0108 spin-0 PsiFormer comparison at step
`29999` and continues both attention routes for 10000 additional KFAC steps
with KFAC damping raised from `1e-3` to `2e-3`.

The completed 0109 beta-20 continuation is intentionally not used as the
restore source.

## Continuation Semantics

- Restore source: 0108 final checkpoints at `qmcjax_ckpt_029999.npz`
- Training config sets `iterations: 40000`
- New steps: 30000 through 39999
- New output root:
  `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000`

The completed 0108 directories are read only as restore sources and are not
used as save paths.

## Variants

| Attention QKV route | Experiment config | Slurm name |
| --- | --- | --- |
| ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp2e3_merge_none_batch4096_iter40000_continue0110.yaml` | `solidnes-0110-ferminet-b10-d2e3-cont10k` |
| fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp2e3_merge_none_batch4096_iter40000_continue0110.yaml` | `solidnes-0110-fused_qkv-b10-d2e3-cont10k` |

## Fixed Settings

- Batch size: 4096
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=10.0`
- KFAC damping: `0.002`
- KFAC norm constraint: `0.001`
- Learning rate: `0.05`
- Spin logging: `log_spin_by_state: true`, `log_every: 1`
- Full S2 matrix observable: `observables_s2: false`

## Submission

Build-only verification passed for both configs before submission. Jobs were
submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.

| Job ID | Attention QKV route | Current/latest state | Node/reason |
| ---: | --- | --- | --- |
| 137472 | ferminet | COMPLETED, exit `0:0`, elapsed `01:45:35` | `gpuh2001` |
| 137473 | fused_qkv | COMPLETED, exit `0:0`, elapsed `01:45:04` | `gpuh2001` |

Slurm plans are under `outputs/slurm_plans/`.

## Result Summary

Both 0110 jobs completed the requested damping-2e-3 continuation steps
`30000..39999`. Ground/excited energies are sorted per step from the appended
`energy_matrix.npy` stream.

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin | Last1000 spin state0 | Last1000 spin state1 | Last1000 |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.39309419 | -75.18852078 | 5.566726 +/- 1.380652 | 5.599056 +/- 1.416943 | 0.006235 | 0.005364 | 0.007106 | 0.007186 |
| Fused QKV | -75.39636183 | -75.18883065 | 5.647211 +/- 1.410595 | 5.611575 +/- 1.454483 | 0.006337 | 0.004768 | 0.007906 | 0.007162 |

Compared with the completed 0108 beta-10/damping-1e-3 tail:

- FermiNet-QKV last1000 gap changed from `5.626026 eV` to `5.566726 eV`;
  last5000 gap changed from `5.569321 eV` to `5.599056 eV`.
- Fused-QKV last1000 gap changed from `5.482634 eV` to `5.647211 eV`;
  last5000 gap changed from `5.558044 eV` to `5.611575 eV`.
- Last1000 spin decreased for both routes: FermiNet-QKV from `0.008264` to
  `0.006235`, and fused-QKV from `0.007474` to `0.006337`.

Summary artifacts:

- `analysis/0110_combined_result_summary.md`
- `analysis/0110_combined_summary.csv`
- `analysis/0110_combined_summary.json`
- `analysis/0110_ferminet_damp2e3_result_summary.json`
- `analysis/0110_fused_qkv_damp2e3_result_summary.json`
- `analysis/0110_combined_timeseries.csv`
- `analysis/0110_damp1e3_to_damp2e3_timeseries.csv`

Iteration plots:

- `analysis/0110_iteration_plots.md`
- `analysis/0110_damp2e3_energy_gap_spin_rolling_after30000_window1000.png`
- `analysis/0110_damp2e3_gap_rolling_after30000_window1000.png`
- `analysis/0110_damp2e3_energy_gap_spin_rolling_after35000_window1000.png`
- `analysis/0110_damp2e3_gap_rolling_after35000_window1000.png`
- `analysis/0110_damp1e3_to_damp2e3_energy_gap_spin_rolling_after10000_window1000.png`
- `analysis/0110_damp1e3_to_damp2e3_gap_rolling_after10000_window1000.png`
- `analysis/0110_damp1e3_to_damp2e3_energy_gap_spin_rolling_after20000_window1000.png`
- `analysis/0110_damp1e3_to_damp2e3_gap_rolling_after20000_window1000.png`
