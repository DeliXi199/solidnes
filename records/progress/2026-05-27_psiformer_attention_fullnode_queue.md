# 2026-05-27 PsiFormer Attention Full-Node Queue

Task: `tasks/psiformer/0096_psiformer_attention_full_stack/`

Per user direction, pretraining is out of scope and `test` is only for flow
validation. Full training-speed comparisons are queued once per attention
variant with Slurm `--partition amdgpu40g,amdgpu80g`, so whichever partition
opens first can run the job. This is not a 40GB-vs-80GB speed comparison.

Configuration:

- Model: paper-scale PsiFormer PBC, 16 determinants, 4 layers, 4 heads, head dim 64.
- Training: native FermiNet `vmc_overlap`, KFAC, FOLX forward Laplacian, batch4096, 10000 iterations.
- Spin: `spin_penalty=0.0`, `observables_s2=false`; no `s2_matrix.npy` is expected for the speed jobs.
- Hardware request per job: 4 GPUs, 64 CPU cores, exclusive node, 64 GB Slurm memory request.

Cancelled redundant fixed-partition jobs:

- `131692`: `amdgpu40g`, upstream FermiNet attention.
- `131693`: `amdgpu40g`, fused-QKV attention.
- `131694`: `amdgpu80g`, upstream FermiNet attention.
- `131695`: `amdgpu80g`, fused-QKV attention.

Completed active jobs:

- `131735`: `amdgpu40g,amdgpu80g`, upstream FermiNet attention, `fullnode_anygpu_ferminet_b4096_i10000`.
- `131736`: `amdgpu40g,amdgpu80g`, fused-QKV attention, `fullnode_anygpu_fused_qkv_b4096_i10000`.

Replaced jobs:

- `131697`: cancelled combined-partition upstream/FermiNet 2000-step job.
- `131698`: cancelled combined-partition fused-QKV 2000-step job.

Completion:

- `131735` ran on `amdgpu40g/gpu006`, completed in `01:26:37`, exit `0:0`.
- `131736` ran on `amdgpu40g/gpu006`, completed in `01:26:50`, exit `0:0`.
- Both wrote 10000 training rows and finite final diagnostic arrays.
- Runtime metadata reports 0.514879 s/iteration for upstream FermiNet attention and 0.518262 s/iteration for fused-QKV attention.
- Fused-QKV was about 0.657% slower end-to-end in this native KFAC/FOLX training workload.

Primary comparison artifact:
`tasks/psiformer/0096_psiformer_attention_full_stack/results/validation/psiformer_fullnode_10k_attention_comparison.md`.
