# PsiFormer Attention Build Benchmark

Date: 2026-05-27, Asia/Shanghai

## Summary

Task `0094` created the first PsiFormer/self-attention task bundle under:

```text
tasks/psiformer/0094_psiformer_attention_build_benchmark/
```

SolidNES now supports FermiNet PsiFormer configs with a configurable attention
implementation:

```text
model.attention.implementation: auto | ferminet | fused_qkv
```

The `fused_qkv` implementation follows the LapNet-style self-attention
optimization: Q, K, and V are projected with one fused matrix and then split.
The represented self-attention function is the same as FermiNet's upstream
three-projection implementation. DeepQMC's PsiFormer route and LapNet's
attention/LapTuple implementation were inspected as references.

## Local Results

Build-only config passed for:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_attention_build_benchmark.yaml
```

Key config fields:

```text
network_type: psiformer
objective: vmc_overlap
states: 2
batch_size: 4096
laplacian: folx
psiformer_attention_implementation: auto
```

Local CPU forward benchmark:

```text
walkers: 64
warmup: 2
repeats: 8
upstream median: 0.009195 s
fused-QKV median: 0.009120 s
median speedup: 1.008x
fused-QKV mean speedup: 0.863x because of one CPU timing outlier
max_abs_logabs_delta: 0.0
max_abs_sign_delta: 0.0
```

Interpretation:

```text
The fused implementation is numerically identical on this benchmark. This CPU
timing is kept as a correctness check, not as a production-policy input.
```

## GPU Results

The first GPU benchmark job `131634` reached `amdgpu40g/gpu005` but failed
before model execution because the benchmark script set `JAX_PLATFORMS=gpu`;
this CUDA JAX environment expects `cuda` or an unset platform. The benchmark
script now normalizes `--platform gpu` to `cuda`, and the Slurm planner was
fixed so an explicit wrapper-level `SOLIDNES_GPU_ALLOW_TEST=1` submission can
override the default `test` blocked partition.

The corrected lightweight benchmark was submitted to the test GPU partition as
Slurm job `131644`:

```text
node: test001
gpu: 1x RTX 4090
cpu: 8 cores
walltime request: 00:15:00
elapsed: about 24 seconds
plan: tasks/psiformer/0094_psiformer_attention_build_benchmark/outputs/slurm_plans/plan_gpu_benchmark_test_cuda_submit.json
logs: tasks/psiformer/0094_psiformer_attention_build_benchmark/logs/slurm/solidnes-0094-psiformer-attn-test_131644.log
```

GPU forward benchmark:

```text
walkers: 256
warmup: 5
repeats: 20
upstream median: 0.000454 s
fused-QKV median: 0.000432 s
median speedup: 1.051x
mean speedup: 1.052x
max_abs_logabs_delta: 0.0
max_abs_sign_delta: 0.0
all_finite: true
```

Interpretation:

```text
The fused-QKV implementation preserves the upstream PsiFormer self-attention
output on the tested CPU and GPU batches. The measured GPU forward speedup is
modest but positive. After the later GPU-only production clarification, the
default `auto` policy resolves directly to fused-QKV; use the explicit
`ferminet` option only for controls.
```
