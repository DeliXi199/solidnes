# Reference Notes: PsiFormer Attention Implementation

## Sources Checked

- `external/ferminet/ferminet/psiformer.py`
- `external/deepqmc/src/deepqmc/conf/ansatz/psiformer.yaml`
- `external/deepqmc/src/deepqmc/gnn/update_features.py`
- `external/LapNet/lapnet/networks/transformer_blocks.py`
- `external/LapNet/lapnet/networks/lapnet.py`

## Implementation Choice

SolidNES keeps FermiNet's PsiFormer model contract and adds a configurable
attention implementation switch:

```text
model.attention.implementation: auto | ferminet | fused_qkv
model.attention.kernel_gpu: jax | pallas | reference
```

The default SolidNES PsiFormer PBC config uses `auto`, and `auto` resolves to
`fused_qkv`. This is intentionally GPU-production oriented: subsequent
PsiFormer calculations are expected to run on GPU, so the default should not be
controlled by CPU microbenchmark behavior. The default GPU attention core is
`jax`, because the x64 KFAC/FOLX runs validated here avoid the Pallas
custom-call reverse-mode path and its compile fragility. The explicit `pallas`
and `reference` kernels remain available for ablation runs.

The fused path follows LapNet's self-attention optimization: compute Q, K, and
V with one projection, then split the result into the three streams. The
represented function is the same as FermiNet's three-projection self-attention,
but with fewer projection calls.

DeepQMC's PsiFormer route uses Haiku `MultiHeadAttention` inside a residual
GNN update. It is useful as a shape and residual-connection reference, but it
does not provide the PBC/FermiNet training integration used by SolidNES.

LapNet also contains a sparse LapTuple-aware attention path for forward
Laplacian propagation. SolidNES does not port LapJAX into FermiNet here; the
safe first speed improvement is the fused QKV projection while preserving the
existing FOLX Laplacian path.

## Training Smoke Result

Task 0095 validated the native PsiFormer training path on the `test` GPU
partition. The `auto` attention policy resolved to `fused_qkv` on GPU, and KFAC
registration showed the fused `qkv_w` block instead of separate `q_w`, `k_w`,
and `v_w` blocks.

The short batch512 full-training comparison was effectively tied:
upstream FermiNet attention took 36.855 s/iteration, while fused-QKV took
37.075 s/iteration (`0.994x` speedup). This means the five-step native training
smoke is dominated by KFAC/local-energy/MCMC/JIT overhead rather than the
attention projection. The task 0094 forward-only GPU benchmark still showed
the expected attention-level benefit: fused-QKV matched outputs exactly and
improved median forward time by `1.051x` at 256 walkers.

## Current Recommendation

Keep `auto -> fused_qkv` as the default policy for production. The explicit
`ferminet` option remains available for ablation/control runs. Across the
completed x64 GPU validation runs, fused-QKV was not faster in the two-state
excited-state workload and was about 2% slower than FermiNet-shaped Q/K/V on
the same node class, but it produced smoother excited-state trajectories with
smaller late-window gap fluctuations. The 30k ground-state comparison also
converged cleanly with fused-QKV and showed no stability regression.

The practical default is therefore:

- use `model.attention.implementation: fused_qkv` or `auto` for production;
- use `model.attention.kernel_gpu: jax` for x64 KFAC/FOLX runs;
- keep `ferminet` only when an explicit upstream-shaped Q/K/V control is
  needed.
