# Task 0112: Attention QKV Spin Beta-10 Damping-1e-3 Learning-Rate Sweep

Task root:

```text
tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000
```

This task branches from the two completed 0108 spin-0 PsiFormer runs:

- `ferminet`
- `fused_qkv`

Both continuation routes restore from the 0108 `qmcjax_ckpt_029999.npz`, keep
the DeepQMC-style spin penalty at `beta=10.0`, keep KFAC `damping=0.001`, set
`iterations: 40000`, and therefore run 10000 additional steps: 30000 through
39999. The sweep tests base learning rates `0.02`, `0.01`, and `0.005`, with
both attention routes for each value.

Fixed-ground is explicit-only and is not used.

Submitted jobs:

| Job ID | Learning rate | Route | Current/latest status |
| ---: | ---: | --- | --- |
| 138677 | `0.02` | ferminet | COMPLETED |
| 138678 | `0.02` | fused_qkv | COMPLETED |
| 138679 | `0.01` | ferminet | COMPLETED |
| 138680 | `0.01` | fused_qkv | COMPLETED |
| 138681 | `0.005` | ferminet | COMPLETED |
| 138682 | `0.005` | fused_qkv | COMPLETED |

Dry-run and submitted Slurm plans are written under
`tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/outputs/slurm_plans/`.

Analysis outputs:

- `tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_comparison.md`
- `tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_summary.csv`
- `tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis/0112_lr_sweep_combined_timeseries.csv`
