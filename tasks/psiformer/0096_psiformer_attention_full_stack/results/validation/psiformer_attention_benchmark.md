# PsiFormer Attention Benchmark

Experiment: `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_forward_benchmark.yaml`

| Implementation | Mean s | Median s | Speedup mean | Logabs delta | Finite |
| --- | ---: | ---: | ---: | ---: | --- |
| ferminet | 0.000821 | 0.000821 | 1.000 | 0.000e+00 | True |
| fused_qkv | 0.000814 | 0.000814 | 1.008 | 0.000e+00 | True |
