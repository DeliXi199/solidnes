# 0111 Attention QKV Spin Beta-10 Damping Sweep Continuation

This task branches from the completed 0108 spin-0 PsiFormer comparison at step
`29999` and continues both attention routes for 10000 additional KFAC steps
for each damping value in `{3e-3, 5e-3, 1e-2}`.

The completed 0109 beta-20 continuation and completed 0110 damping-2e-3
continuation are intentionally not used as restore sources.

## Continuation Semantics

- Restore source: 0108 final checkpoints at `qmcjax_ckpt_029999.npz`
- Training config sets `iterations: 40000`
- New steps: 30000 through 39999
- New output root:
  `tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000`

The completed 0108 directories are read only as restore sources and are not
used as save paths.

## Variants

| Damping | Attention QKV route | Experiment config | Slurm name |
| ---: | --- | --- | --- |
| `0.003` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp3e3_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-ferminet-b10-d3e3-cont10k` |
| `0.003` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp3e3_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-fused_qkv-b10-d3e3-cont10k` |
| `0.005` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp5e3_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-ferminet-b10-d5e3-cont10k` |
| `0.005` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp5e3_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-fused_qkv-b10-d5e3-cont10k` |
| `0.01` | ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e2_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-ferminet-b10-d1e2-cont10k` |
| `0.01` | fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp1e2_merge_none_batch4096_iter40000_continue0111.yaml` | `solidnes-0111-fused_qkv-b10-d1e2-cont10k` |

## Fixed Settings

- Batch size: 4096
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=10.0`
- KFAC norm constraint: `0.001`
- Learning rate: `0.05`
- Spin logging: `log_spin_by_state: true`, `log_every: 1`
- Full S2 matrix observable: `observables_s2: false`

## Submission

Build-only verification should pass for all six configs before submission.
Jobs are submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.

Initial submissions `138133`--`138138` were cancelled before allocation because
the previous default flexible GPU partition set still included the disabled
`h20` partition. The submitter and planner defaults now block `h20`, and the
active submissions below request only `h200,amdgpu80g,amdgpu40g`.

| Job ID | Damping | Attention QKV route | Latest status | Notes |
| ---: | ---: | --- | --- | --- |
| 138139 | `0.003` | ferminet | COMPLETED | Finished on `gpuh2001` |
| 138140 | `0.003` | fused_qkv | COMPLETED | Finished on `gpu002` |
| 138141 | `0.005` | ferminet | COMPLETED | Finished on `gpuh2001` |
| 138142 | `0.005` | fused_qkv | TIMEOUT | Last valid checkpoint is `qmcjax_ckpt_036433.npz`; job later hung after a CUDA graph teardown error |
| 138143 | `0.01` | ferminet | COMPLETED | Finished on `gpu002` |
| 138144 | `0.01` | fused_qkv | RUNNING | Running on `gpu002` when checked at 2026-06-05 09:40 CST |

## Resume Submission

The interrupted `0.005` fused-QKV branch is resumed from the latest valid
checkpoint written by job `138142`:

- Restore checkpoint directory:
  `tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/runs/damp5e3/fused_qkv_merge_none/results/checkpoints`
- Latest checkpoint: `qmcjax_ckpt_036433.npz`
- Resume config:
  `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_damp5e3_merge_none_batch4096_iter40000_continue0111_resume36433.yaml`
- New output directory:
  `tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/runs/damp5e3/fused_qkv_merge_none_resume36433`
- New steps expected from restore: 36434 through 39999

| Job ID | Damping | Attention QKV route | State after submit | Reason after submit |
| ---: | ---: | --- | --- | --- |
| 138403 | `0.005` | fused_qkv resume from 36433 | CANCELLED | Cancelled before allocation to resubmit with `h200` included |
| 138411 | `0.005` | fused_qkv resume from 36433 | PENDING | Resources; requested `h200,amdgpu80g,amdgpu40g` |

Slurm plans are under `outputs/slurm_plans/`.

## Damping Sweep Comparison

Generated on 2026-06-05 after all damping jobs completed:

- Summary:
  `analysis/0111_damping_sweep_comparison.md`
- Combined time series:
  `analysis/0111_damping_sweep_combined_timeseries.csv`
- Final-window summary:
  `analysis/0111_damping_sweep_summary.csv`
- FermiNet full-window energy/gap/spin plot:
  `analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.png`
- FermiNet zoomed energy/gap/spin plot:
  `analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.png`
- FermiNet full-window gap plot:
  `analysis/0111_damping_sweep_ferminet_gap_rolling_after30000_window1000.png`
- FermiNet zoomed gap plot:
  `analysis/0111_damping_sweep_ferminet_gap_rolling_after35000_window1000.png`
- Fused-QKV full-window energy/gap/spin plot:
  `analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.png`
- Fused-QKV zoomed energy/gap/spin plot:
  `analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.png`
- Fused-QKV full-window gap plot:
  `analysis/0111_damping_sweep_fused_qkv_gap_rolling_after30000_window1000.png`
- Fused-QKV zoomed gap plot:
  `analysis/0111_damping_sweep_fused_qkv_gap_rolling_after35000_window1000.png`

All comparison statistics use continuation steps 30000 through 39999 and a
trailing 1000-step mean. The `damping=0.005` fused-QKV route is assembled from
the original timed-out segment and the successful resume from step 36433.
