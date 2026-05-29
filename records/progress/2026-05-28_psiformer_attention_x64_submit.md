# 2026-05-28 PsiFormer Attention X64 Resubmission

Per user request, the two completed PsiFormer attention 10000-step comparison
jobs were reconfigured for 64-bit precision and resubmitted without replacing
the existing speed/TF32 results.

Added x64 model configs:

```text
configs/model/psiformer_pbc_paper_attention_ferminet_x64.yaml
configs/model/psiformer_pbc_paper_attention_fused_qkv_x64.yaml
```

Both keep the same paper-scale PsiFormer topology as the previous 10k jobs, but
set `psiformer.tf32: false`.

Added x64 experiment configs:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_b4096_iter10000.yaml
configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_b4096_iter10000.yaml
```

Both set:

```text
runtime.precision_profile: fp64
runtime.x64_enabled: true
training batch_size: 4096
training iterations: 10000
states: 2
objective: vmc_overlap
optimizer: KFAC
laplacian: FOLX
spin_penalty: 0.0
observables_s2: false
pretrain_iterations: 0
```

Build-only checks passed for both configs and confirmed:

```text
precision_profile: fp64
x64_enabled: True
psiformer_tf32: False
```

Submitted jobs:

| Variant | Job | Partition request | State at submit check | Resources |
| --- | ---: | --- | --- | --- |
| upstream/FermiNet attention x64 | `131952` | `amdgpu40g,amdgpu80g` | `RUNNING` on `gpu006` | 4 GPUs, 64 CPUs, exclusive |
| fused-QKV attention x64 | `131953` | `amdgpu40g,amdgpu80g` | `PENDING (Resources)` | 4 GPUs, 64 CPUs, exclusive |

The running upstream x64 job log confirms:

```text
JAX_ENABLE_X64=1
CUDA_VISIBLE_DEVICES=0,1,2,3
jax=0.10.1
jax_devices=[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_b4096_iter10000.yaml
```

The upstream x64 job has entered graph construction/training startup. The log
currently shows a FOLX dtype-promotion warning at the PsiFormer spin-feature
concatenate, but no traceback; keep this in mind when interpreting x64 wall time
if the warning persists for both variants.

## FOLX Spin-Feature Concatenate Fix

The warning was reproduced with a minimal x64 FOLX function: the original pattern
of concatenating a coordinate-dependent feature with a fixed spin feature forces
FOLX to concatenate an `int64` sparse mask with an `int32` filled mask. The
SolidNES runtime PsiFormer patch now replaces the upstream input concatenate
with a value-preserving form:

```text
spin_features = spin_features + ae_features[..., :1] * 0.0
features = concatenate(ae_features, spin_features)
```

The spin channel value is unchanged, but it enters the concatenate with a
compatible zero-derivative sparse mask, so FOLX no longer needs the dense
materialization fallback for this operation in the minimal repro.

Build-only checks passed for the two clean rerun configs:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_folxfix_b4096_iter10000.yaml
configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_folxfix_b4096_iter10000.yaml
```

Jobs `131952` and `131953` were cancelled/replaced to keep the formal x64
comparison clean. New submitted jobs:

| Variant | Job | Partition request | State at submit check | Resources |
| --- | ---: | --- | --- | --- |
| fused-QKV attention x64 FOLX-fix | `131974` | `amdgpu40g,amdgpu80g` | `RUNNING` on `gpu006` | 4 GPUs, 64 CPUs, exclusive |
| upstream/FermiNet attention x64 FOLX-fix | `131975` | `amdgpu40g,amdgpu80g` | `PENDING (Resources)` | 4 GPUs, 64 CPUs, exclusive |

Log check for running job `131974` at 6:32 elapsed showed the usual GPU compile
message after burn-in, but no repeat of the earlier FOLX concatenate warning.

Follow-up after completion:

```text
python scripts/validation/plot_psiformer_fullnode_attention_iterations.py
python scripts/validation/plot_psiformer_fullnode_state_gap.py
```

The plotting scripts currently target the original non-x64 run names, so add
x64 variants or command-line run selection before final x64 analysis.

## Slow-Compile Mitigation Iteration

The x64 FOLX-fix jobs exposed a separate XLA compile issue after burn-in:

- `132019`/plain fused-QKV x64 reached `Completed burn-in MCMC steps` and then
  stalled on `input_reduce_fusion.77`.
- `132051` reduced the state-specific FOLX kinetic to only the needed diagonal
  scalar state outputs; this changed the slow module to `input_reduce_fusion.22`
  but did not remove the slow compile.
- XLA LLVM parallelism probes (`132034`, `132038`, `132072`) failed immediately
  on `nvlink fatal: Input file ... newer than toolkit (129 vs 128)`, so the
  cluster CUDA linker path is not usable as a fix.

Code changes now in place:

- `ferminet_pbc_hamiltonian._state_specific_folx_kinetic` differentiates only
  each state's scalar diagonal output instead of differentiating a full
  state-output vector and taking its diagonal.
- `ferminet.loss.make_energy_overlap_loss` accepts
  `max_local_energy_vmap_batch_size`, so only the FOLX local-energy vmap is
  chunked. The KFAC log-wavefunction vmap remains dense and KFAC
  auto-registration keeps the repeated-dense/scale-and-shift blocks.
- PsiFormer attention has a configurable GPU kernel:
  `pallas`, `reference`, or `jax`. The pure-JAX kernel avoids the Pallas
  custom-call reverse-mode issue in KFAC.

Probe outcomes so far:

| Job | Config | Result |
| ---: | --- | --- |
| `132091` | batch4096, x64, local-energy vmap=128, Pallas attention | Initial energy compiled and ran, but first KFAC update failed with Pallas reverse-mode linearization. |
| `132106` | batch4096, x64, local-energy vmap=128, reference attention | Completed 2 steps, but first KFAC update still emitted `jit__fun` slow compile taking 3m08s. |
| `132167` | test partition, batch512, x64, local-energy vmap=64, pure-JAX attention | Completed 2 steps on 2x4090 with no `Very slow compile` and no Pallas linearization error. |
| `132113` | full-node batch4096, x64, local-energy vmap=128, pure-JAX attention | Completed on `amdgpu80g/gpu002` in `00:06:51`, exit `0:0`; log grep found no `Very slow compile`, Pallas linearization, traceback, or nvlink errors. |

The `132113` full-node probe passed, so the prepared 10000-step x64 comparison
configs were submitted:

| Variant | Config | Job | Partition request | State at submit check | Resources |
| --- | --- | ---: | --- | --- | --- |
| upstream/FermiNet attention x64, pure-JAX attention kernel | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml` | `132575` | `amdgpu40g,amdgpu80g` | `PENDING (Priority)` | 4 GPUs, 64 CPUs, exclusive |
| fused-QKV attention x64, pure-JAX attention kernel | `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml` | `132574` | `amdgpu40g,amdgpu80g` | `PENDING (Resources)` | 4 GPUs, 64 CPUs, exclusive |

Both jobs use the same user-facing training settings: two states, no spin
penalty, no S2 observable, batch4096, 10000 iterations, KFAC, FOLX,
`precision_profile: fp64`, `JAX_ENABLE_X64=1`, `psiformer_tf32=false`, and
`max_local_energy_vmap_batch_size: 128`.
