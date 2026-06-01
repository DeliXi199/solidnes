# Task 0104: DeepQMC Spin-Penalty Alignment

This slice records the DeepQMC-style spin-penalty alignment and the required
SLURM-submitted GPU smoke.

| Task | Status | Key Result |
| --- | --- | --- |
| 0104 | completed | SLURM job 135738 completed on `amdgpu40g/gpu004`; finite train stats plus energy/overlap/S2 arrays; `energy_matrix.npy == bare_energy_matrix.npy` confirms loss-level spin penalty diagnostics. |

Task root:

```text
tasks/excited_state_nesvmc/0104_deepqmc_spin_penalty_alignment_gpu_smoke/
```

Configs:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_beta10_smoke_batch128_iter2.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_smoke_batch128_iter2.yaml
```
