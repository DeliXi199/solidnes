# Task 0105: DeepQMC State-Specific Spin Alignment

This task validates the final spin-penalty semantics after replacing the loss
operator with a state-specific local S² estimator matching DeepQMC
`evaluate_spin`.

| Task | Status | Key Result |
| --- | --- | --- |
| 0105 | completed | SLURM job 135764 completed on `amdgpu40g/gpu004`; finite train stats plus energy/overlap/S2 arrays; loss uses state-specific DeepQMC local S² estimator. |

Task root:

```text
tasks/excited_state_nesvmc/0105_deepqmc_spin_state_specific_gpu_smoke/
```

Config:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_state_specific_beta10_smoke_batch128_iter2.yaml
```
