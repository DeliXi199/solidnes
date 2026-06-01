# Tasks 0094-0103: PsiFormer And DeepQMC Alignment

This slice covers the PsiFormer attention work and the DeepQMC-aligned
excited-state method selection.

| Task | Summary | Key Result |
| --- | --- | --- |
| 0094 | Configurable PsiFormer attention implementation and forward benchmarks. | GPU `auto` resolves to `fused_qkv`; fused-QKV matched upstream outputs and improved forward median time on RTX 4090. |
| 0095 | Native PsiFormer training-path smoke and attention comparison. | Full-training b512 timing was effectively tied; training bottleneck was outside attention projection. |
| 0096 | No-pretrain paper-scale attention full-stack validation. | Speed-profile 10000-step comparison completed; later superseded for method selection by 0097-0103. |
| 0097 | DeepQMC-aligned independent-state route for the two attention variants. | KFAC shape issue fixed by using one full parameter pytree per state; both 10000-step retries completed. |
| 0098 | GPU smoke for DeepQMC alignment changes. | Donation-aliasing issue fixed by making merged leaves value-equal but buffer-independent. |
| 0099 | Small diagonal-path speed benchmark. | Timing excludes compile/first-step warmup and compares diagonal-on/off. |
| 0100 | Production-like 4GPU diagonal-path speed benchmark. | Stable batch4096 step time improved from 2.165 s to 1.268 s, about 1.71x faster. |
| 0101 | 4GPU 200-step merge-key sweep. | `merge_none` and `merge_embed` were faster than `merge_layers`; merge remained a comparison question. |
| 0102 | Final non-merge alignment validation. | 4GPU diagonal path was 1.72x faster than full-matrix control; KFAC state-axis validation passed. |
| 0103 | 10000-step attention x merge-key comparison. | Supports the final source-code default: fused-QKV with `merge_keys: []`; non-empty merge keys remain optional comparison branches. |

Current source-code milestone:

```text
34d6574 Set no-merge excited-state mainline
916bcc4 Record no-merge excited-state milestone
records/progress/2026-06-01_excited_state_no_merge_mainline.md
```

Task-local details:

```text
tasks/psiformer/0094_psiformer_attention_build_benchmark/README.md
tasks/psiformer/0095_psiformer_native_training_smoke/README.md
tasks/psiformer/0096_psiformer_attention_full_stack/README.md
tasks/psiformer/0097_deepqmc_aligned_excited_state/README.md
tasks/psiformer/0098_deepqmc_alignment_gpu_smoke/README.md
tasks/psiformer/0099_deepqmc_alignment_speed_benchmark/README.md
tasks/psiformer/0100_deepqmc_alignment_4gpu_speed_benchmark/README.md
tasks/psiformer/0101_deepqmc_merge_keys_4gpu_sweep/README.md
tasks/psiformer/0102_deepqmc_nonmerge_alignment_final_validation/README.md
tasks/psiformer/0103_attention_merge_keys_4gpu_10000/README.md
```
