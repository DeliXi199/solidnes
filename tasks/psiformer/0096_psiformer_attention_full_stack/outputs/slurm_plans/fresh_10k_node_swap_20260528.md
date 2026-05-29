# Fresh 10k Node-Swap Submissions

Submitted on 2026-05-28 CST for a clean method/node comparison. Both jobs start
from fresh initialization, use x64/fp64, batch size 4096, 4 GPUs, 64 CPUs,
10,000 VMC iterations, pure-JAX attention, no spin penalty, and no S2
observables.

| Job ID | Method | Node | Partition | Config | Run directory |
| --- | --- | --- | --- | --- | --- |
| 132947 | FermiNet-shaped separate Q/K/V | gpu002 | amdgpu80g | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_gpu002_ferminet_x64_fresh_b4096_iter10000_levmap128_jaxattn.yaml` | `tasks/psiformer/0096_psiformer_attention_full_stack/runs/gpu002_ferminet_x64_fresh_b4096_i10000_levmap128_jaxattn` |
| 132948 | fused-QKV | gpu005 | amdgpu40g | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_gpu005_fused_qkv_x64_fresh_b4096_iter10000_levmap128_jaxattn.yaml` | `tasks/psiformer/0096_psiformer_attention_full_stack/runs/gpu005_fused_qkv_x64_fresh_b4096_i10000_levmap128_jaxattn` |

Queue context at submission:

- `132947` is pending for `gpu002`; `hold-after-132860-gpu002` (`132945`) is
  currently occupying that node.
- `132948` is pending for `gpu005`; continuation job `132861` is currently
  occupying that node.
