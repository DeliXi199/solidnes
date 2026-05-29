# Task 0096: PsiFormer Attention Full Stack

## Purpose

Add configurable self-attention implementations to the SolidNES/FermiNet
PsiFormer path and validate them on GPU for both excited-state and ground-state
diamond calculations.

## Source Integration

The production integration lives in:

- `src/solidnes/backends/ferminet_psiformer_attention.py`
- `src/solidnes/backends/ferminet_adapter.py`
- `src/solidnes/excited_states/ferminet_pbc_adapter.py`
- `scripts/backends/run_ferminet_train.py`

Use model YAML:

```yaml
model:
  attention:
    implementation: auto      # auto resolves to fused_qkv
    kernel_gpu: jax           # default for x64 KFAC/FOLX stability
```

Supported implementations:

- `fused_qkv`: recommended default. Computes Q, K, and V with one fused
  projection and then splits the result.
- `ferminet`: upstream-shaped control. Keeps separate Q, K, and V projections
  while using the SolidNES FOLX-aware attention core.

Supported GPU attention cores:

- `jax`: recommended for x64 KFAC/FOLX production runs.
- `pallas`: available for attention-only ablations.
- `reference`: CPU/reference fallback.

## Recommended Configs

Production/default:

- `configs/model/psiformer_pbc_paper.yaml`
- `configs/model/psiformer_pbc_paper_attention_fused_qkv_x64_jaxattn.yaml`

Control:

- `configs/model/psiformer_pbc_paper_attention_ferminet_x64_jaxattn.yaml`

Completed 30k ground-state configs:

- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_ground_state_anygpu_fused_qkv_x64_b4096_iter30000_levmap128_jaxattn_vmcchunk.yaml`
- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_ground_state_anygpu_ferminet_x64_b4096_iter30000_levmap128_jaxattn_vmcchunk.yaml`

Completed 10k excited-state comparison configs:

- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml`
- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_ferminet_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml`

## Completed Result Summary

Excited-state x64 10k 2x2 comparison:

- Same-node runtime: fused-QKV was about 2% slower than FermiNet Q/K/V.
- Stability: fused-QKV showed smaller last-1000 gap fluctuation on both node
  classes and no late root flipping.
- Result summary:
  `results/validation/x64_10k_2x2_method_partition/psiformer_x64_10k_2x2_analysis.md`

Ground-state x64 30k VMCChunk comparison:

- FermiNet Q/K/V job `133248`: `0.5165 s/iter`, last1000 energy
  `-75.414326 +/- 0.030321 Ha`.
- fused-QKV job `133249`: `0.4804 s/iter`, last1000 energy
  `-75.416211 +/- 0.029899 Ha`.
- Runtime comparison is confounded by different node classes, but fused-QKV
  converged cleanly and showed no stability regression.
- Result summary:
  `results/validation/ground_state_30k_vmcchunk/ground_state_30k_vmcchunk_analysis.md`

## Current Recommendation

Keep both implementations in the project. Use fused-QKV by default for new
GPU PsiFormer runs because it is at least as accurate in the completed
ground-state validation and gives smoother excited-state trajectories. Use
FermiNet Q/K/V only as a control/ablation path.
