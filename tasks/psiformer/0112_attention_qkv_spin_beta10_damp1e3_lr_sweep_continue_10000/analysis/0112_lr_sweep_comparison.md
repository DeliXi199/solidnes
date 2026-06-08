# 0112 Learning-Rate Sweep Comparison

Compared continuation steps 30000 through 39999. All runs restore from the completed 0108 step-29999 checkpoints and keep `damping=0.001`, `spin_penalty=10.0`, and `norm_constraint=0.001`; only the KFAC base learning rate changes.

Statistics below use the final trailing 1000-step window.

| Route | Learning rate | Rows | E0 mean Ha | E1 mean Ha | Gap eV | Spin mean | pmove | step s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ferminet | 0.005 | 10000 | -75.395425 | -75.190116 | 5.5868 | 0.007533 | 0.5349 | 0.599 |
| ferminet | 0.01 | 10000 | -75.394124 | -75.188076 | 5.6069 | 0.006323 | 0.5349 | 1.617 |
| ferminet | 0.02 | 10000 | -75.390807 | -75.186913 | 5.5482 | 0.006474 | 0.5348 | 1.624 |
| fused_qkv | 0.005 | 10000 | -75.395088 | -75.189788 | 5.5865 | 0.006437 | 0.5349 | 1.640 |
| fused_qkv | 0.01 | 10000 | -75.394766 | -75.186358 | 5.6711 | 0.005347 | 0.5349 | 0.610 |
| fused_qkv | 0.02 | 10000 | -75.391179 | -75.187081 | 5.5538 | 0.005777 | 0.5349 | 1.645 |

## Generated Plots

- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_gap_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_ferminet_gap_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_gap_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_fused_qkv_gap_rolling_after35000_window1000.svg`
