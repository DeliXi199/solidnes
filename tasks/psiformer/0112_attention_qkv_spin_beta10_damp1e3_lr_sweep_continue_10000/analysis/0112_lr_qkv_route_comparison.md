# 0112 Learning-Rate QKV Route Comparison

Compared `ferminet` QKV and `fused_qkv` at each learning rate. Plots use trailing 1000-step means and keep each learning rate in a separate figure.

## after30000

| Learning rate | ΔE0 fused-ferminet Ha | ΔE1 fused-ferminet Ha | Δgap fused-ferminet eV | Δspin fused-ferminet |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | -0.000373 | -0.000168 | 0.0056 | -0.000697 |
| 0.01 | -0.000641 | 0.001719 | 0.0642 | -0.000976 |
| 0.005 | 0.000337 | 0.000328 | -0.0003 | -0.001096 |

## after20000

| Learning rate | ΔE0 fused-ferminet Ha | ΔE1 fused-ferminet Ha | Δgap fused-ferminet eV | Δspin fused-ferminet |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | -0.000373 | -0.000168 | 0.0056 | -0.000697 |
| 0.01 | -0.000641 | 0.001719 | 0.0642 | -0.000976 |
| 0.005 | 0.000337 | 0.000328 | -0.0003 | -0.001096 |

## Generated Plots

- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr2e2_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr2e2_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr2e2_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr2e2_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr1e2_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr1e2_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr1e2_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr1e2_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr5e3_energy_gap_spin_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr5e3_energy_gap_spin_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr5e3_gap_rolling_after30000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after30000_lr5e3_gap_rolling_after30000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr2e2_energy_gap_spin_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr2e2_energy_gap_spin_rolling_after20000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr2e2_gap_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr2e2_gap_rolling_after20000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr1e2_energy_gap_spin_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr1e2_energy_gap_spin_rolling_after20000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr1e2_gap_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr1e2_gap_rolling_after20000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr5e3_energy_gap_spin_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr5e3_energy_gap_spin_rolling_after20000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr5e3_gap_rolling_after20000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_qkv_route_comparison_after20000_lr5e3_gap_rolling_after20000_window1000.svg`
