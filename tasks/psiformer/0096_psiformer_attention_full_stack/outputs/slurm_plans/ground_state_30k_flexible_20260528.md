# Ground-State 30k Flexible Queue Submissions

Submitted on 2026-05-28 CST using the project GPU submitter in flexible queue
mode. These jobs do not request a fixed node; they queue on
`amdgpu40g,amdgpu80g,h200` and let Slurm assign a node. They were initially
submitted to `amdgpu40g,amdgpu80g`, then updated with `scontrol update` to add
the regular `h200` partition.

Common settings:

- Target: ground state only
- Objective: `vmc`
- States: `0`
- Precision: x64/fp64
- Model: paper-scale PsiFormer
- Batch size: `4096`
- Iterations: `30000`
- GPUs/CPUs: `4 GPU + 64 CPU`
- Attention kernel: pure JAX
- Pretraining: disabled
- Spin penalty: `0.0`
- S2 observable: disabled
- Submission policy: `queue_mode=flexible`, `ReqNodeList=(null)`
- Partition request after update: `amdgpu40g,amdgpu80g,h200`

| Job ID | Method | Partition request | Config | Run directory |
| --- | --- | --- | --- | --- |
| 132957 | FermiNet-shaped separate Q/K/V | `amdgpu40g,amdgpu80g,h200` | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_ground_state_anygpu_ferminet_x64_b4096_iter30000_levmap128_jaxattn.yaml` | `tasks/psiformer/0096_psiformer_attention_full_stack/runs/ground_state_anygpu_ferminet_x64_b4096_i30000_levmap128_jaxattn` |
| 132958 | fused-QKV | `amdgpu40g,amdgpu80g,h200` | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_ground_state_anygpu_fused_qkv_x64_b4096_iter30000_levmap128_jaxattn.yaml` | `tasks/psiformer/0096_psiformer_attention_full_stack/runs/ground_state_anygpu_fused_qkv_x64_b4096_i30000_levmap128_jaxattn` |

Plan JSON:

- `tasks/psiformer/0096_psiformer_attention_full_stack/outputs/slurm_plans/plan_ground_state_ferminet_x64_b4096_i30000_flexible_submit.json`
- `tasks/psiformer/0096_psiformer_attention_full_stack/outputs/slurm_plans/plan_ground_state_fused_qkv_x64_b4096_i30000_flexible_submit.json`
