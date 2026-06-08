# 0109 Attention QKV Spin-20 Continuation

This task continues the completed 0108 spin-0 PsiFormer comparison for 10000
additional KFAC steps with the DeepQMC-style spin penalty increased from
`beta=10.0` to `beta=20.0`.

Fixed-ground is intentionally not part of this task.

## Continuation Semantics

- Restore source: 0108 final checkpoints at `qmcjax_ckpt_029999.npz`
- FermiNet restore returns `t_init = 30000`
- Training config sets `iterations: 40000`
- New steps: 30000 through 39999
- New output root: `tasks/psiformer/0109_attention_qkv_spin20_continue_10000`

The completed 0108 directories are read only as restore sources and are not
used as save paths.

## Variants

| Attention QKV route | Experiment config | Slurm name |
| --- | --- | --- |
| ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta20_merge_none_batch4096_iter40000_continue0109.yaml` | `solidnes-0109-ferminet-b20-cont10k` |
| fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta20_merge_none_batch4096_iter40000_continue0109.yaml` | `solidnes-0109-fused_qkv-b20-cont10k` |

## Fixed Settings

- Batch size: 4096
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=20.0`
- Spin logging: `log_spin_by_state: true`, `log_every: 1`
- Full S2 matrix observable: `observables_s2: false`

## Submission

Build-only verification passed for both configs before submission. Jobs were
submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.

| Job ID | Attention QKV route | Current/latest state | Node/reason |
| ---: | --- | --- | --- |
| 137016 | ferminet | COMPLETED, exit `0:0`, elapsed `01:44:36` | `gpuh2001` |
| 137015 | fused_qkv | COMPLETED, exit `0:0`, elapsed `04:50:50` | `gpu004` |

Slurm plans are under `outputs/slurm_plans/`.

## Partial Result: Fused QKV

`137015` completed the requested beta-20 continuation steps `30000..39999`.
Ground/excited energies are sorted per step from `energy_matrix.npy`.

| Window | Ground (Ha) | Excited (Ha) | Gap (eV) | Spin | Spin state0 | Spin state1 | |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last1000 | -75.36276706 | -75.16250815 | 5.449322 +/- 1.571569 | 0.005222 | 0.004046 | 0.006398 | 0.010906 |
| Last5000 | -75.36186146 | -75.15969682 | 5.501180 +/- 1.560627 | 0.005846 | 0.004967 | 0.006725 | 0.009994 |

Compared with the completed 0108 fused-QKV beta-10 tail:

- Last1000 gap changed from `5.482634 eV` to `5.449322 eV`.
- Last5000 gap changed from `5.558044 eV` to `5.501180 eV`.
- Last1000 spin changed from `0.007474` to `0.005222`.

Summary JSON:

- `analysis/0109_fused_qkv_beta20_result_summary.json`

Iteration plots:

- `analysis/0109_fused_qkv_beta20_iteration_plots.md`
- `analysis/0109_fused_qkv_beta10_to_beta20_iteration_plots_after10000.md`
- `analysis/0109_fused_qkv_beta10_to_beta20_energy_gap_spin_rolling_after10000_window1000.png`
- `analysis/0109_fused_qkv_beta10_to_beta20_gap_rolling_after10000_window1000.png`
- `analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after30000_window1000.png`
- `analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after35000_window1000.png`
- `analysis/0109_fused_qkv_beta10_to_beta20_energy_gap_spin_rolling_after20000_window1000.png`
- `analysis/0109_fused_qkv_beta10_to_beta20_gap_rolling_after20000_window1000.png`

## Result: FermiNet QKV

`137016` completed the requested beta-20 continuation steps `30000..39999`.
Ground/excited energies are sorted per step from `energy_matrix.npy`.

| Window | Ground (Ha) | Excited (Ha) | Gap (eV) | Spin | Spin state0 | Spin state1 | |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last1000 | -75.36439099 | -75.16074979 | 5.542265 +/- 1.528610 | 0.006483 | 0.005318 | 0.007649 | 0.009430 |
| Last5000 | -75.36192374 | -75.15888923 | 5.525755 +/- 1.568047 | 0.006392 | 0.005144 | 0.007640 | 0.008435 |

Compared with the completed 0108 FermiNet-QKV beta-10 tail:

- Last1000 gap changed from `5.626026 eV` to `5.542265 eV`.
- Last5000 gap changed from `5.569321 eV` to `5.525755 eV`.
- Last1000 spin changed from `0.008264` to `0.006483`.

Summary JSON:

- `analysis/0109_ferminet_qkv_beta20_result_summary.json`

Iteration plots:

- `analysis/0109_ferminet_qkv_beta20_iteration_plots.md`
- `analysis/0109_ferminet_qkv_beta20_energy_gap_spin_rolling_after30000_window1000.png`
- `analysis/0109_ferminet_qkv_beta20_energy_gap_spin_rolling_after35000_window1000.png`
- `analysis/0109_ferminet_qkv_beta10_to_beta20_energy_gap_spin_rolling_after10000_window1000.png`
- `analysis/0109_ferminet_qkv_beta10_to_beta20_gap_rolling_after10000_window1000.png`
- `analysis/0109_ferminet_qkv_beta10_to_beta20_energy_gap_spin_rolling_after20000_window1000.png`
- `analysis/0109_ferminet_qkv_beta10_to_beta20_gap_rolling_after20000_window1000.png`
