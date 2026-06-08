# Task 0110: Attention QKV Spin Beta-10 Damping 2e-3 Continuation

Task root:

```text
tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000
```

This task branches from the two completed 0108 spin-0 PsiFormer runs:

- `ferminet`
- `fused_qkv`

Both continuation jobs restore from the 0108 `qmcjax_ckpt_029999.npz`, keep
the DeepQMC-style spin penalty at `beta=10.0`, raise KFAC damping from `1e-3`
to `2e-3`, set `iterations: 40000`, and therefore run 10000 additional steps:
30000 through 39999.

Fixed-ground is explicit-only and is not used.

Submitted jobs:

| Job ID | Route | Current/latest status |
| ---: | --- | --- |
| 137472 | ferminet | COMPLETED, exit `0:0`, elapsed `01:45:35`, node `gpuh2001` |
| 137473 | fused_qkv | COMPLETED, exit `0:0`, elapsed `01:45:04`, node `gpuh2001` |

Dry-run and submitted Slurm plans are written under
`tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/outputs/slurm_plans/`.

Result summary:

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin | Last1000 |S offdiag| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.39309419 | -75.18852078 | 5.566726 +/- 1.380652 | 5.599056 +/- 1.416943 | 0.006235 | 0.007186 |
| Fused QKV | -75.39636183 | -75.18883065 | 5.647211 +/- 1.410595 | 5.611575 +/- 1.454483 | 0.006337 | 0.007162 |

Compared with the completed 0108 beta-10/damping-1e-3 tail, damping 2e-3
kept the gap in the same `5.56-5.65 eV` band while lowering last1000 spin for
both routes.

Main artifacts:

- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_combined_result_summary.md`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_iteration_plots.md`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp2e3_energy_gap_spin_rolling_after30000_window1000.png`
- `tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000/analysis/0110_damp1e3_to_damp2e3_energy_gap_spin_rolling_after10000_window1000.png`
