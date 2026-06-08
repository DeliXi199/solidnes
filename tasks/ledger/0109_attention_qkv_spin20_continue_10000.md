# Task 0109: Attention QKV Spin-20 Continuation

Task root:

```text
tasks/psiformer/0109_attention_qkv_spin20_continue_10000
```

This task continues the two completed 0108 spin-0 PsiFormer runs:

- `ferminet`
- `fused_qkv`

Both continuation jobs restore from 0108 `qmcjax_ckpt_029999.npz`, increase the
DeepQMC-style spin penalty from `beta=10.0` to `beta=20.0`, set
`iterations: 40000`, and therefore run 10000 additional steps: 30000 through
39999.

Fixed-ground is explicit-only and is not used.

Submitted jobs:

| Job ID | Route | Current/latest status |
| ---: | --- | --- |
| 137016 | ferminet | COMPLETED, exit `0:0`, elapsed `01:44:36`, node `gpuh2001` |
| 137015 | fused_qkv | COMPLETED, exit `0:0`, elapsed `04:50:50` |

Build-only verification passed for both configs before submission. Dry-run and
submitted Slurm plans were written under
`tasks/psiformer/0109_attention_qkv_spin20_continue_10000/outputs/slurm_plans/`.

Partial result for completed fused-QKV job `137015`:

| Window | Ground (Ha) | Excited (Ha) | Gap (eV) | Spin | Spin state0 | Spin state1 | |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last1000 | -75.36276706 | -75.16250815 | 5.449322 +/- 1.571569 | 0.005222 | 0.004046 | 0.006398 | 0.010906 |
| Last5000 | -75.36186146 | -75.15969682 | 5.501180 +/- 1.560627 | 0.005846 | 0.004967 | 0.006725 | 0.009994 |

Compared with the completed 0108 fused-QKV beta-10 tail, beta 20 lowered the
last1000 spin from `0.007474` to `0.005222`, while the last1000 gap changed
from `5.482634 eV` to `5.449322 eV`.

Iteration plot artifacts:

- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta20_iteration_plots.md`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_iteration_plots_after10000.md`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_energy_gap_spin_rolling_after10000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_gap_rolling_after10000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after30000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta20_energy_gap_spin_rolling_after35000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_energy_gap_spin_rolling_after20000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_fused_qkv_beta10_to_beta20_gap_rolling_after20000_window1000.png`

Result for completed FermiNet-QKV job `137016`:

| Window | Ground (Ha) | Excited (Ha) | Gap (eV) | Spin | Spin state0 | Spin state1 | |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last1000 | -75.36439099 | -75.16074979 | 5.542265 +/- 1.528610 | 0.006483 | 0.005318 | 0.007649 | 0.009430 |
| Last5000 | -75.36192374 | -75.15888923 | 5.525755 +/- 1.568047 | 0.006392 | 0.005144 | 0.007640 | 0.008435 |

Compared with the completed 0108 FermiNet-QKV beta-10 tail, beta 20 lowered
the last1000 spin from `0.008264` to `0.006483`, while the last1000 gap changed
from `5.626026 eV` to `5.542265 eV`.

FermiNet-QKV iteration plot artifacts:

- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta20_iteration_plots.md`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta20_energy_gap_spin_rolling_after30000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta20_energy_gap_spin_rolling_after35000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta10_to_beta20_energy_gap_spin_rolling_after10000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta10_to_beta20_gap_rolling_after10000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta10_to_beta20_energy_gap_spin_rolling_after20000_window1000.png`
- `tasks/psiformer/0109_attention_qkv_spin20_continue_10000/analysis/0109_ferminet_qkv_beta10_to_beta20_gap_rolling_after20000_window1000.png`
