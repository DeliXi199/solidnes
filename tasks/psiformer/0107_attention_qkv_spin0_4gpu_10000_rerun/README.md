# 0107 Attention QKV Spin-0 4GPU 10000-Step Re-run

This task re-submits the two PsiFormer attention QKV handling routes from
task 0106 after the DeepQMC-style spin-penalty JVP reuse audit.

Fixed-ground is intentionally not part of this task. Project policy remains:
do not select or submit fixed-ground unless the user explicitly asks for
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
  `spin_state_0`/`spin_state_1` CSV columns from the loss-level estimator
- Full S2 matrix observable: `observables_s2: false`
- Slurm partitions: `amdgpu40g,amdgpu80g,h200` through the flexible 4GPU policy

## Variants

| Attention QKV route | Experiment config | Slurm name |
| --- | --- | --- |
| fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_merge_none_batch4096_iter10000_rerun0107.yaml` | `solidnes-0107-fused_qkv-spin0` |
| ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_merge_none_batch4096_iter10000_rerun0107.yaml` | `solidnes-0107-ferminet-spin0` |

## Submission Notes

The numerical training config matches task 0106. Only experiment names, Slurm
names, and output paths were changed so the running task 0106 jobs are not
touched.

## Submitted Jobs

| Job ID | Attention QKV route | State after submit | Reason after submit |
| ---: | --- | --- | --- |
| 136006 | ferminet | PENDING | Resources |
| 136007 | fused_qkv | PENDING | Priority |

Submitted on 2026-06-02 through `scripts/slurm/submit_ferminet_gpu_smoke.sh`
after build-only verification and DeepQMC-alignment validation passed.
