# 0109 fused-QKV beta-20 iteration plots

Plots use the same 1000-step rolling-mean convention as the 0108 analysis. Energies are sorted per step from `energy_matrix.npy`; spin uses `spin_state_0` and `spin_state_1` from `train_stats.csv`.

## Artifacts

- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after30000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after35000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_energy_gap_spin_rolling_after20000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_gap_rolling_after20000_window1000.png`

## Numeric anchors

| Window | Gap (eV) | Spin | Spin state0 | Spin state1 |
| --- | ---: | ---: | ---: | ---: |
| Last1000 | 5.449322 +/- 1.571569 | 0.005222 | 0.004046 | 0.006398 |
| Last5000 | 5.501180 +/- 1.560627 | 0.005846 | 0.004967 | 0.006725 |
