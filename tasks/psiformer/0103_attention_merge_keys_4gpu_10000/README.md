# 0103 Attention x Merge Keys 4GPU 10000-Step Sweep

This task compares the two 0096 PsiFormer attention implementations against
four independent-state parameter sharing choices after the non-merge DeepQMC
alignment work was fixed.

## Fixed Settings

- Batch size: 4096
- Iterations: 10000
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Diagonal fast paths: MCMC trace, local energy, overlap JVP all enabled
- Component profiling: disabled
- Step timing: enabled
- Slurm partitions: `amdgpu40g,amdgpu80g,h200`

## Submitted Jobs

| Job ID | Attention | Merge keys | Slurm name |
| ---: | --- | --- | --- |
| 134841 | fused_qkv | none | solidnes-0103-fused_qkv-none |
| 134842 | fused_qkv | layers | solidnes-0103-fused_qkv-layers |
| 134843 | fused_qkv | layers/embed | solidnes-0103-fused_qkv-embed |
| 134844 | fused_qkv | envelope, jastrow, layers/embed | solidnes-0103-fused_qkv-low_level |
| 134845 | ferminet | none | solidnes-0103-ferminet-none |
| 134846 | ferminet | layers | solidnes-0103-ferminet-layers |
| 134847 | ferminet | layers/embed | solidnes-0103-ferminet-embed |
| 134848 | ferminet | envelope, jastrow, layers/embed | solidnes-0103-ferminet-low_level |

All jobs were submitted with the project GPU submitter using flexible queue mode,
so pending jobs keep the combined partition target
`amdgpu40g,amdgpu80g,h200`.

## Evaluation Plan

After all runs finish, compare:

- final and tail-window state energies
- final and tail-window gap
- symmetric overlap matrix
- overlap penalty matrix
- energy-weighted variance
- acceptance / pmove
- stable step time excluding compilation and early initialization
- any state collapse or root flipping signs
