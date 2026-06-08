# 0106 Attention QKV Spin-0 4GPU 10000-Step Comparison

This task submits the two current PsiFormer attention QKV handling routes for
the DeepQMC-aligned two-state excited-state method.

Fixed-ground is intentionally not part of this task. Project policy from this
point: do not select or submit fixed-ground unless the user explicitly asks for
`fixed-ground`.

## Fixed Settings

- Batch size: 4096
- Iterations: 10000
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Diagonal fast paths: MCMC trace, local energy, overlap JVP all enabled
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=10.0`
- Spin target: total spin `S=0` (`S^2=0`) for both optimized states
- Spin logging: `log_spin_by_state: true`, `log_every: 1`, per-state
  `spin_state_0`/`spin_state_1` CSV columns from the DeepQMC-style penalty
  estimator
- Full S2 matrix observable: `observables_s2: false`
- Slurm partitions: `amdgpu40g,amdgpu80g,h200`

## Variants

| Attention QKV route | Experiment config | Slurm name |
| --- | --- | --- |
| fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_merge_none_batch4096_iter10000.yaml` | `solidnes-0106-fused_qkv-spin0` |
| ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_merge_none_batch4096_iter10000.yaml` | `solidnes-0106-ferminet-spin0` |

## Submitted Jobs

| Job ID | Attention QKV route | State after submit | Reason after submit |
| ---: | --- | --- | --- |
| 135878 | fused_qkv | CANCELLED | Replaced before start to remove full S2 matrix diagnostics |
| 135879 | ferminet | CANCELLED | Replaced before start to remove full S2 matrix diagnostics |
| 135932 | fused_qkv | PENDING | Priority |
| 135931 | ferminet | PENDING | Resources |

The corrected jobs `135932` and `135931` were submitted on 2026-06-01 through
`scripts/slurm/submit_ferminet_gpu_smoke.sh` after build-only verification,
DeepQMC-alignment validation, and dry-run plans passed.

## Expected Spin Outputs

- `train_stats.csv` logs `spin`, `spin_penalty`, `spin_state_0`, and
  `spin_state_1` every step.
- No full `s2_matrix.npy` observable is written; per-state spin comes directly
  from the loss-level DeepQMC-style spin estimator.
- `bare_energy_matrix.npy` keeps the unpenalized energy side by side with the
  spin-penalized loss route.
