# Task 0111: Attention QKV Spin Beta-10 Damping Sweep Continuation

Task root:

```text
tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000
```

This task branches from the two completed 0108 spin-0 PsiFormer runs:

- `ferminet`
- `fused_qkv`

Both continuation routes restore from the 0108 `qmcjax_ckpt_029999.npz`, keep
the DeepQMC-style spin penalty at `beta=10.0`, set `iterations: 40000`, and
therefore run 10000 additional steps: 30000 through 39999. The sweep tests
KFAC damping values `3e-3`, `5e-3`, and `1e-2`, with both attention routes for
each value.

Fixed-ground is explicit-only and is not used.

Submitted jobs:

Initial submissions `138133`--`138138` were cancelled before allocation because
the previous default flexible GPU partition set still included the disabled
`h20` partition. The active resubmissions below were planned after adding `h20`
to the blocked partition list, so their partition request is
`h200,amdgpu80g,amdgpu40g`.

| Job ID | Damping | Route | Current/latest status |
| ---: | ---: | --- | --- |
| 138139 | `0.003` | ferminet | COMPLETED |
| 138140 | `0.003` | fused_qkv | COMPLETED |
| 138141 | `0.005` | ferminet | COMPLETED |
| 138142 | `0.005` | fused_qkv | TIMEOUT; last valid checkpoint `qmcjax_ckpt_036433.npz` |
| 138143 | `0.01` | ferminet | COMPLETED |
| 138144 | `0.01` | fused_qkv | RUNNING on `gpu002` when checked at 2026-06-05 09:40 CST |

Resume job:

| Job ID | Damping | Route | Current/latest status |
| ---: | ---: | --- | --- |
| 138403 | `0.005` | fused_qkv, resumed from `qmcjax_ckpt_036433.npz` | CANCELLED before allocation to resubmit with `h200` included |
| 138411 | `0.005` | fused_qkv, resumed from `qmcjax_ckpt_036433.npz` | PENDING, reason `Resources`; requested `h200,amdgpu80g,amdgpu40g` |

Dry-run and submitted Slurm plans are written under
`tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/outputs/slurm_plans/`.
