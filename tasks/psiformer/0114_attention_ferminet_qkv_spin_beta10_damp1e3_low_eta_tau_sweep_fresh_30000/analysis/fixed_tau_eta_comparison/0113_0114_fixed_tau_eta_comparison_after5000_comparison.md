# 0113/0114 Fixed-Tau Eta0 Comparison (after5000)

Data combine tasks 0113 and 0114. Mean plots use trailing 1000-step means after step 5000.
Fluctuation plots use the same trailing 1000-step window for variance/std/absolute one-step changes.

Final-window statistics use steps 29000-29999.

| Tau | eta0 | Rows | E0 mean Ha | E0 std | E1 mean Ha | E1 std | Gap eV | Gap std | Spin mean | Spin std | EWVar mean | EWVar std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.001 | 30000 | -75.345198 | 0.050604 | -75.132318 | 0.048569 | 5.7928 | 1.9209 | 0.010527 | 0.000758 | 0.001197 | 0.000451 |
| 10000 | 0.005 | 30000 | -75.381118 | 0.041407 | -75.174438 | 0.041925 | 5.6241 | 1.6289 | 0.008031 | 0.000782 | 0.000852 | 0.000310 |
| 10000 | 0.01 | 30000 | -75.385754 | 0.038196 | -75.183812 | 0.039380 | 5.4951 | 1.5117 | 0.008298 | 0.000750 | 0.000739 | 0.000273 |
| 10000 | 0.02 | 30000 | -75.387870 | 0.035996 | -75.182838 | 0.036058 | 5.5792 | 1.3807 | 0.008607 | 0.000667 | 0.000654 | 0.000343 |
| 15000 | 0.001 | 30000 | -75.349243 | 0.049576 | -75.136587 | 0.050118 | 5.7867 | 1.8974 | 0.009906 | 0.000757 | 0.001267 | 0.000502 |
| 15000 | 0.005 | 30000 | -75.381305 | 0.040733 | -75.179274 | 0.041296 | 5.4975 | 1.6145 | 0.008655 | 0.000743 | 0.000811 | 0.000248 |
| 15000 | 0.01 | 30000 | -75.387676 | 0.038861 | -75.180327 | 0.035913 | 5.6423 | 1.4395 | 0.009135 | 0.000728 | 0.000683 | 0.000264 |
| 15000 | 0.02 | 30000 | -75.387942 | 0.037798 | -75.184550 | 0.036460 | 5.5346 | 1.4335 | 0.009389 | 0.000670 | 0.000689 | 0.000267 |
| 20000 | 0.001 | 30000 | -75.346158 | 0.051958 | -75.137843 | 0.047208 | 5.6686 | 1.8711 | 0.010941 | 0.000863 | 0.001237 | 0.000378 |
| 20000 | 0.005 | 30000 | -75.384725 | 0.040045 | -75.177279 | 0.039244 | 5.6449 | 1.5364 | 0.008132 | 0.000710 | 0.000751 | 0.000254 |
| 20000 | 0.01 | 30000 | -75.384974 | 0.040098 | -75.182021 | 0.037370 | 5.5226 | 1.4780 | 0.007952 | 0.000748 | 0.000746 | 0.000283 |
| 20000 | 0.02 | 30000 | -75.389235 | 0.037751 | -75.180730 | 0.037973 | 5.6737 | 1.4663 | 0.007153 | 0.000720 | 0.000697 | 0.000382 |

## Drift And Step Changes

Delta columns compare the first 1000-step window after 5000 against the final 1000-step window.

| Tau | eta0 | dE0 Ha | dE1 Ha | dGap eV | dSpin | mean abs dE0/step | mean abs dE1/step | mean abs dGap/step | mean abs dSpin/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.001 | -0.086977 | -0.257322 | -4.6353 | -0.015290 | 0.053210 | 0.052243 | 2.0597 | 0.000551 |
| 10000 | 0.005 | -0.063469 | -0.081652 | -0.4948 | -0.014707 | 0.044253 | 0.045726 | 1.7541 | 0.000491 |
| 10000 | 0.01 | -0.059855 | -0.075667 | -0.4303 | -0.012734 | 0.041603 | 0.042678 | 1.6650 | 0.000476 |
| 10000 | 0.02 | -0.062492 | -0.062957 | -0.0126 | -0.012027 | 0.038750 | 0.040009 | 1.5358 | 0.000472 |
| 15000 | 0.001 | -0.085326 | -0.138314 | -1.4419 | -0.018380 | 0.052777 | 0.051976 | 2.0048 | 0.000548 |
| 15000 | 0.005 | -0.070202 | -0.062469 | 0.2104 | -0.011886 | 0.045967 | 0.046493 | 1.7803 | 0.000522 |
| 15000 | 0.01 | -0.065265 | -0.065697 | -0.0117 | -0.012290 | 0.041518 | 0.039959 | 1.5877 | 0.000485 |
| 15000 | 0.02 | -0.063363 | -0.073540 | -0.2769 | -0.010601 | 0.040761 | 0.039057 | 1.5783 | 0.000492 |
| 20000 | 0.001 | -0.090250 | -0.289501 | -5.4219 | -0.017062 | 0.054692 | 0.050555 | 2.0115 | 0.000536 |
| 20000 | 0.005 | -0.066986 | -0.067978 | -0.0270 | -0.012788 | 0.043597 | 0.041723 | 1.6763 | 0.000464 |
| 20000 | 0.01 | -0.066103 | -0.061580 | 0.1231 | -0.013878 | 0.042530 | 0.038961 | 1.6059 | 0.000501 |
| 20000 | 0.02 | -0.062455 | -0.072065 | -0.2615 | -0.014051 | 0.040912 | 0.040195 | 1.6132 | 0.000486 |

## Generated Plots

- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_std_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_std_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_var_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_var_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_abs_delta_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_rolling_abs_delta_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_ewvar_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau10000_ewvar_rolling_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_std_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_std_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_var_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_var_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_abs_delta_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_rolling_abs_delta_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_ewvar_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau15000_ewvar_rolling_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_std_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_std_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_var_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_var_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_abs_delta_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_rolling_abs_delta_after5000_window1000.svg`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_ewvar_rolling_after5000_window1000.png`
- `/data/home/yihaoxu/research/projects/solidnes/tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/0113_0114_fixed_tau_eta_comparison_tau20000_ewvar_rolling_after5000_window1000.svg`
