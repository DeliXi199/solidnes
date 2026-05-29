# PsiFormer Full-Node Speed Matrix

Task root: `tasks/psiformer/0096_psiformer_attention_full_stack`

Active queued jobs use paper-scale PsiFormer, no pretraining, native FermiNet
`vmc_overlap`, KFAC, FOLX forward Laplacian, batch 4096, 10000 iterations,
4 GPUs, and 64 CPU cores. Training parameters are aligned with the prior
native FermiNet excited-state batch4096/KFAC/FOLX settings where applicable.
Spin penalty and S2 observables are disabled.

The initial four fixed-partition submissions were redundant for the intended
queue strategy and were cancelled:

- `131692`: `amdgpu40g`, upstream FermiNet attention.
- `131693`: `amdgpu40g`, fused-QKV attention.
- `131694`: `amdgpu80g`, upstream FermiNet attention.
- `131695`: `amdgpu80g`, fused-QKV attention.

The first combined-partition 2000-step pair was also cancelled and replaced
with the 10000-step pair:

- `131697`: `amdgpu40g,amdgpu80g`, upstream FermiNet attention, 2000 steps.
- `131698`: `amdgpu40g,amdgpu80g`, fused-QKV attention, 2000 steps.

The active comparison is submitted once per attention variant with combined
Slurm partitions, so either `amdgpu40g` or `amdgpu80g` can start the job.

| Variant | Partition | Attention | Batch | Iterations | GPUs | CPUs | Job ID | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `fullnode_anygpu_ferminet_b4096_i10000` | `amdgpu40g,gpu006` | `ferminet` | 4096 | 10000 | 4 | 64 | `131735` | completed `0:0`, 01:26:37 |
| `fullnode_anygpu_fused_qkv_b4096_i10000` | `amdgpu40g,gpu006` | `fused_qkv` | 4096 | 10000 | 4 | 64 | `131736` | completed `0:0`, 01:26:50 |

Comparison result: `131735` completed at 0.514879 s/iteration, while `131736`
completed at 0.518262 s/iteration. Fused-QKV was about 0.657% slower in the
end-to-end native KFAC/FOLX training workload, despite the earlier forward-only
benchmark speedup.

## X64 Resubmission

Per user request, the two 10000-step attention jobs were reconfigured and
resubmitted with `runtime.x64_enabled=true`, `precision_profile=fp64`, and
`psiformer.tf32=false`. All other training parameters remain matched to the
completed speed comparison: no pretraining, native FermiNet `vmc_overlap`,
KFAC, FOLX, batch4096, 10000 iterations, two states, no spin penalty, and no
S2 observables.

| Variant | Partition | Attention | Precision | Batch | Iterations | GPUs | CPUs | Job ID | Status at submit check |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `fullnode_anygpu_ferminet_x64_b4096_i10000` | `amdgpu40g,amdgpu80g` | `ferminet` | x64, TF32 off | 4096 | 10000 | 4 | 64 | `131952` | running on `amdgpu40g/gpu006` |
| `fullnode_anygpu_fused_qkv_x64_b4096_i10000` | `amdgpu40g,amdgpu80g` | `fused_qkv` | x64, TF32 off | 4096 | 10000 | 4 | 64 | `131953` | pending `(Resources)` |

X64 plan files:

- `plan_fullnode_anygpu_ferminet_x64_b4096_i10000_dryrun.json`
- `plan_fullnode_anygpu_ferminet_x64_b4096_i10000_submit.json`
- `plan_fullnode_anygpu_fused_qkv_x64_b4096_i10000_dryrun.json`
- `plan_fullnode_anygpu_fused_qkv_x64_b4096_i10000_submit.json`

## X64 FOLX Spin-Feature Fix Resubmission

The first x64 upstream job `131952` reached FOLX graph construction and showed a
PsiFormer spin-feature concatenate dtype-mask warning. SolidNES now patches the
PsiFormer input layer so the fixed spin channel enters the concatenate through a
zero-derivative coordinate-dependent term, preserving values while avoiding the
FOLX x64 sparse-index fallback. Jobs `131952` and `131953` were cancelled and
replaced with clean `x64_folxfix` run directories.

| Variant | Partition | Attention | Precision | Batch | Iterations | GPUs | CPUs | Job ID | Status at submit check |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `fullnode_anygpu_fused_qkv_x64_folxfix_b4096_i10000` | `amdgpu40g,amdgpu80g` | `fused_qkv` | x64, TF32 off, FOLX spin concat fix | 4096 | 10000 | 4 | 64 | `131974` | running on `amdgpu40g/gpu006` |
| `fullnode_anygpu_ferminet_x64_folxfix_b4096_i10000` | `amdgpu40g,amdgpu80g` | `ferminet` | x64, TF32 off, FOLX spin concat fix | 4096 | 10000 | 4 | 64 | `131975` | pending `(Resources)` |

X64 FOLX-fix plan files:

- `plan_fullnode_anygpu_fused_qkv_x64_folxfix_b4096_i10000_submit.json`
- `plan_fullnode_anygpu_ferminet_x64_folxfix_b4096_i10000_submit.json`

## X64 Slow-Compile Fix Candidates

The x64 FOLX-fix full-batch jobs reached burn-in and then hit a GPU XLA slow
compile in the first energy path (`input_reduce_fusion.*`). The current
candidate fix keeps all user-facing training parameters aligned
(batch4096, two states, no spin penalty, no S2 observable, KFAC, FOLX) and only
changes compiler-facing implementation details:

- state-specific PBC FOLX kinetic differentiates one scalar diagonal state at a
  time;
- only the FOLX local-energy vmap is chunked (`max_local_energy_vmap_batch_size:
  128`);
- PsiFormer attention can use a pure-JAX kernel for x64 KFAC compatibility.

| Variant | Partition | Attention | Kernel | Batch | Iterations | GPUs | CPUs | Job ID | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `fullnode_anygpu_fused_qkv_x64_attnfix_b4096_i2_diagkin_levmap128` | `amdgpu40g/gpu006` | `fused_qkv` | `pallas` | 4096 | 2 | 4 | 64 | `132091` | initial energy passed; KFAC update failed on Pallas reverse-mode linearization |
| same output via env override | `amdgpu40g/gpu006` | `fused_qkv` | `reference` | 4096 | 2 | 4 | 64 | `132106` | completed 2 steps; first KFAC update emitted `jit__fun` slow compile taking 3m08s |
| `test_jaxattn_x64_smoke` | `test/test001` | `fused_qkv` | `jax` | 512 | 2 | 2 | 64 | `132167` | completed 2 steps, no `Very slow compile` warning |
| `fullnode_anygpu_fused_qkv_x64_attnfix_b4096_i2_diagkin_levmap128_jaxattn` | `amdgpu80g/gpu002` | `fused_qkv` | `jax` | 4096 | 2 | 4 | 64 | `132113` | completed `0:0`, 00:06:51; no slow-compile, Pallas, traceback, or nvlink errors in log grep |

The `132113` full-node probe passed, so the matched 10000-step x64 comparison
jobs were submitted with the same user-facing training parameters
(batch4096, two states, no spin penalty, no S2 observable, KFAC, FOLX,
`max_local_energy_vmap_batch_size: 128`) and pure-JAX attention kernels.

| Variant | Partition request | Attention | Kernel | Precision | Batch | Iterations | GPUs | CPUs | Job ID | Status at submit check |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `fullnode_anygpu_ferminet_x64_attnfix_b4096_i10000_levmap128_jaxattn` | `amdgpu40g,amdgpu80g` | `ferminet` | `jax` | x64, TF32 off | 4096 | 10000 | 4 | 64 | `132575` | pending `(Priority)` |
| `fullnode_anygpu_fused_qkv_x64_attnfix_b4096_i10000_levmap128_jaxattn` | `amdgpu40g,amdgpu80g` | `fused_qkv` | `jax` | x64, TF32 off | 4096 | 10000 | 4 | 64 | `132574` | pending `(Resources)` |

10000-step x64 comparison configs:

- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml`
- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml`
