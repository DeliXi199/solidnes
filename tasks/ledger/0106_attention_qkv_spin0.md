# Task 0106: Attention QKV Spin-0 10000-Step Comparison

Task root:

```text
tasks/psiformer/0106_attention_qkv_spin0_4gpu_10000
```

This task submits the two current PsiFormer attention QKV handling routes for
the DeepQMC-aligned two-state `vmc_overlap` method:

- `fused_qkv`
- upstream-shaped `ferminet` Q/K/V

Fixed-ground is explicit-only and was not used.

Both jobs use:

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
| 135878 | fused_qkv | CANCELLED before start; superseded by no-S2-matrix logging |
| 135879 | ferminet | CANCELLED before start; superseded by no-S2-matrix logging |
| 135932 | fused_qkv | PENDING, Priority |
| 135931 | ferminet | PENDING, Resources |

Build-only verification passed for both configs before submission, and dry-run
SLURM plans were written under `outputs/slurm_plans/`.

Alignment correction:

- Full `s2_matrix.npy` observables are disabled to stay closer to DeepQMC.
- Per-state spin is written as `spin_state_0` and `spin_state_1` in
  `train_stats.csv`, using the same loss-level state-specific S2 estimator that
  supplies the spin penalty.
