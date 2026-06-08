# Task 0107: Attention QKV Spin-0 10000-Step Re-run

Task root:

```text
tasks/psiformer/0107_attention_qkv_spin0_4gpu_10000_rerun
```

This task re-submits the two current PsiFormer attention QKV handling routes
from task 0106 after the DeepQMC-alignment audit:

- `fused_qkv`
- upstream-shaped `ferminet` Q/K/V

Fixed-ground is explicit-only and is not used.

Both jobs use the same numerical settings as task 0106:

```text
states: 2
merge_keys: []
spin_penalty: 10.0
log_spin_by_state: true
observables_s2: false
log_every: 1
iterations: 10000
batch_size: 4096
gpus/cpus: 4 GPU / 64 CPU
```

Submitted jobs:

| Job ID | Route | Status after submit |
| ---: | --- | --- |
| 136006 | ferminet | PENDING, Resources |
| 136007 | fused_qkv | PENDING, Priority |

Build-only verification passed for both 0107 configs before submission, and
dry-run SLURM plans were written under `outputs/slurm_plans/`.
