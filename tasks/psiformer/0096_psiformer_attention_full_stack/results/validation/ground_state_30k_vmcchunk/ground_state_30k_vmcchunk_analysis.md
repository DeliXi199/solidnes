# PsiFormer ground-state 30k VMCChunk analysis

Completed jobs:

| method | job | node/partition | slurm elapsed | runtime s/iter | final EW mean (Ha) | last1000 energy mean ± std (Ha) | last1000 ewvar | last1000 pmove |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| FermiNet Q/K/V | 133248 | gpu006/amdgpu40g | 04:18:46 | 0.5165 | -75.423091 | -75.414326 ± 0.030321 | 0.000879292 | 0.5348 |
| fused-QKV | 133249 | gpu002/amdgpu80g | 04:00:59 | 0.4804 | -75.417251 | -75.416211 ± 0.029899 | 0.000833818 | 0.5350 |

Notes:

- The earlier non-vmcchunk jobs 132957/132958 were cancelled and are not used for the final comparison.
- These two completed jobs ran on different node classes, so runtime comparison is confounded by node/partition.
- Both jobs reached step 29999 with 30000 rows.
