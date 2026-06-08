# Task 0108: Attention QKV Spin-0 Continuation

Task root:

```text
tasks/psiformer/0108_attention_qkv_spin0_continue_20000
```

This task continues the two completed 0107 spin-0 PsiFormer runs:

- `fused_qkv`
- upstream-shaped `ferminet` Q/K/V

Both continuation jobs restore from 0107 `qmcjax_ckpt_009999.npz`, set
`iterations: 30000`, and therefore run 20000 additional steps: 10000 through
29999.

Fixed-ground is explicit-only and is not used.

Submitted jobs:

| Job ID | Route | Final status |
| ---: | --- | --- |
| 136170 | ferminet | COMPLETED, exit `0:0`, elapsed `09:23:02` |
| 136171 | fused_qkv | COMPLETED, exit `0:0`, elapsed `09:30:38` |

Build-only verification passed for both 0108 continuation configs before
submission, and dry-run SLURM plans were written under `outputs/slurm_plans/`.

Launch verification:

- `136170` loaded the 0107 FermiNet `qmcjax_ckpt_009999.npz` checkpoint and
  reached `Step 10000`.
- `136171` was still pending for GPU resources at the first post-submit check.

Result summary:

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin state0 | Last1000 spin state1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.38956332 | -75.18281067 | 5.626026 +/- 1.579027 | 5.569321 +/- 1.574981 | 0.007474 | 0.009054 |
| Fused QKV | -75.38409287 | -75.18260977 | 5.482634 +/- 1.492863 | 5.558044 +/- 1.559539 | 0.006864 | 0.008084 |

Analysis artifacts:

- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_result_summary.md`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after5000_window1000.png`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after10000_window1000.png`
- `tasks/psiformer/0108_attention_qkv_spin0_continue_20000/analysis/0108_combined_energy_gap_spin_rolling_after20000_window1000.png`
