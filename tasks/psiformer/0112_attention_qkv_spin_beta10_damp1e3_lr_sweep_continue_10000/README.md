# 0112 Attention QKV Spin Beta-10 Damping-1e-3 Learning-Rate Sweep

This task branches from the completed 0108 spin-0 PsiFormer comparison at step
`29999` and continues both attention routes for 10000 additional KFAC steps
for each learning-rate value in `{0.02, 0.01, 0.005}`.

The completed 0109 beta-20 continuation, 0110 damping-2e-3 continuation, and
0111 damping sweep are intentionally not used as restore sources.

## Continuation Semantics

- Restore source: 0108 final checkpoints at `qmcjax_ckpt_029999.npz`
- Training config sets `iterations: 40000`
- New steps: 30000 through 39999
- New output root:
  `tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000`

The completed 0108 directories are read only as restore sources and are not
used as save paths.

## Variants

| Learning rate | Attention QKV route | Experiment config | Slurm name |
| ---: | --- | --- | --- |
| `0.02` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_lr2e2_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-ferminet-b10-d1e3-lr2e2-cont10k` |
| `0.02` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp1e3_lr2e2_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-fused_qkv-b10-d1e3-lr2e2-cont10k` |
| `0.01` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_lr1e2_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-ferminet-b10-d1e3-lr1e2-cont10k` |
| `0.01` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp1e3_lr1e2_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-fused_qkv-b10-d1e3-lr1e2-cont10k` |
| `0.005` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_lr5e3_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-ferminet-b10-d1e3-lr5e3-cont10k` |
| `0.005` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp1e3_lr5e3_merge_none_batch4096_iter40000_continue0112.yaml` | `solidnes-0112-fused_qkv-b10-d1e3-lr5e3-cont10k` |

## Fixed Settings

- Batch size: 4096
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=10.0`
- KFAC damping: `0.001`
- KFAC norm constraint: `0.001`
- Learning-rate schedule: same delay/decay as 0108, only base rate changed
- Spin logging: `log_spin_by_state: true`, `log_every: 1`
- Full S2 matrix observable: `observables_s2: false`

## Submission

Build-only verification passed for all six configs before submission. Jobs are
submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.

The partition request is `h200,amdgpu80g,amdgpu40g`; the disabled `h20`
partition is blocked by the submitter defaults.

| Job ID | Learning rate | Attention QKV route | Latest status | Notes |
| ---: | ---: | --- | --- | --- |
| 138677 | `0.02` | ferminet | COMPLETED | Finished on `gpu006` |
| 138678 | `0.02` | fused_qkv | COMPLETED | Finished on `gpu006` |
| 138679 | `0.01` | ferminet | COMPLETED | Finished on `gpu004` |
| 138680 | `0.01` | fused_qkv | COMPLETED | Finished on `gpuh2001` |
| 138681 | `0.005` | ferminet | COMPLETED | Finished on `gpuh2001` |
| 138682 | `0.005` | fused_qkv | COMPLETED | Finished on `gpu006` |

Slurm plans are under `outputs/slurm_plans/`.

## Learning-Rate Sweep Comparison

Generated on 2026-06-06 after all learning-rate jobs completed:

- Summary:
  `analysis/0112_lr_sweep_comparison.md`
- Combined time series:
  `analysis/0112_lr_sweep_combined_timeseries.csv`
- Final-window summary:
  `analysis/0112_lr_sweep_summary.csv`
- FermiNet full-window energy/gap/spin plot:
  `analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.png`
- FermiNet zoomed energy/gap/spin plot:
  `analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.png`
- FermiNet full-window gap plot:
  `analysis/0112_lr_sweep_ferminet_gap_rolling_after30000_window1000.png`
- FermiNet zoomed gap plot:
  `analysis/0112_lr_sweep_ferminet_gap_rolling_after35000_window1000.png`
- Fused-QKV full-window energy/gap/spin plot:
  `analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.png`
- Fused-QKV zoomed energy/gap/spin plot:
  `analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.png`
- Fused-QKV full-window gap plot:
  `analysis/0112_lr_sweep_fused_qkv_gap_rolling_after30000_window1000.png`
- Fused-QKV zoomed gap plot:
  `analysis/0112_lr_sweep_fused_qkv_gap_rolling_after35000_window1000.png`

All comparison statistics use continuation steps 30000 through 39999 and a
trailing 1000-step mean.

## Last-20000-Step Plots

Generated on 2026-06-06 by stitching the corresponding 0108 route from
steps 20000 through 29999 before the 0112 learning-rate branches from steps
30000 through 39999:

- Summary:
  `analysis/0112_lr_sweep_after20000_comparison.md`
- Combined time series:
  `analysis/0112_lr_sweep_after20000_combined_timeseries.csv`
- FermiNet energy/gap/spin plot:
  `analysis/0112_lr_sweep_after20000_ferminet_energy_gap_spin_rolling_after20000_window1000.png`
- FermiNet gap plot:
  `analysis/0112_lr_sweep_after20000_ferminet_gap_rolling_after20000_window1000.png`
- Fused-QKV energy/gap/spin plot:
  `analysis/0112_lr_sweep_after20000_fused_qkv_energy_gap_spin_rolling_after20000_window1000.png`
- Fused-QKV gap plot:
  `analysis/0112_lr_sweep_after20000_fused_qkv_gap_rolling_after20000_window1000.png`

## QKV Route Comparison By Learning Rate

Generated on 2026-06-06 to compare `ferminet` QKV and `fused_qkv` directly at
each fixed learning rate. Each figure overlays the two QKV routes for the same
learning-rate value using the same trailing 1000-step mean style as the 0110
analysis.

- Summary:
  `analysis/0112_lr_qkv_route_comparison.md`
- Final-window route-difference table:
  `analysis/0112_lr_qkv_route_comparison_after30000_summary.csv`
- Last-20000 route-difference table:
  `analysis/0112_lr_qkv_route_comparison_after20000_summary.csv`
- `lr=0.02`, steps 30000-39999:
  `analysis/0112_lr_qkv_route_comparison_after30000_lr2e2_energy_gap_spin_rolling_after30000_window1000.png`
- `lr=0.01`, steps 30000-39999:
  `analysis/0112_lr_qkv_route_comparison_after30000_lr1e2_energy_gap_spin_rolling_after30000_window1000.png`
- `lr=0.005`, steps 30000-39999:
  `analysis/0112_lr_qkv_route_comparison_after30000_lr5e3_energy_gap_spin_rolling_after30000_window1000.png`
- `lr=0.02`, stitched steps 20000-39999:
  `analysis/0112_lr_qkv_route_comparison_after20000_lr2e2_energy_gap_spin_rolling_after20000_window1000.png`
- `lr=0.01`, stitched steps 20000-39999:
  `analysis/0112_lr_qkv_route_comparison_after20000_lr1e2_energy_gap_spin_rolling_after20000_window1000.png`
- `lr=0.005`, stitched steps 20000-39999:
  `analysis/0112_lr_qkv_route_comparison_after20000_lr5e3_energy_gap_spin_rolling_after20000_window1000.png`

Gap-only PNG/SVG companion plots are generated with the same filename stem and
`gap_rolling` in place of `energy_gap_spin_rolling`.
