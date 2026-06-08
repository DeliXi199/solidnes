# 0111 Damping Sweep Comparison

Compared continuation steps 30000 through 39999.  All runs restore from the completed 0108 step-29999 checkpoints and keep `learning_rate=0.05`, `spin_penalty=10.0`, and `norm_constraint=0.001`; only KFAC damping changes.

Statistics below use the final trailing 1000-step window.

| Route | Damping | Rows | E0 mean Ha | E1 mean Ha | Gap eV | Spin mean | pmove | step s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ferminet | 0.002 | 10000 | -75.393094 | -75.188521 | 5.5667 | 0.006235 | 0.5350 | 0.574 |
| ferminet | 0.003 | 10000 | -75.393436 | -75.187637 | 5.6001 | 0.006266 | 0.5351 | 0.599 |
| ferminet | 0.005 | 10000 | -75.393139 | -75.186956 | 5.6105 | 0.005876 | 0.5348 | 0.598 |
| ferminet | 0.01 | 10000 | -75.394862 | -75.188645 | 5.6115 | 0.005690 | 0.5350 | 1.415 |
| fused_qkv | 0.002 | 10000 | -75.396362 | -75.188831 | 5.6472 | 0.006337 | 0.5350 | 0.578 |
| fused_qkv | 0.003 | 10000 | -75.393739 | -75.188958 | 5.5724 | 0.006088 | 0.5350 | 1.442 |
| fused_qkv | 0.005 | 10000 | -75.392194 | -75.187192 | 5.5784 | 0.006676 | 0.5349 | 1.454 |
| fused_qkv | 0.01 | 10000 | -75.394065 | -75.188208 | 5.6016 | 0.005955 | 0.5349 | 1.446 |

## Generated Plots

- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_energy_gap_spin_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_gap_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_ferminet_gap_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_energy_gap_spin_rolling_after35000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_gap_rolling_after35000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis/0111_damping_sweep_fused_qkv_gap_rolling_after35000_window1000.svg`

## Data Notes

- fused_qkv d=0.005 segment 0: energy_matrix records=7274, train_stats rows=6434; using train_stats row count for alignment
