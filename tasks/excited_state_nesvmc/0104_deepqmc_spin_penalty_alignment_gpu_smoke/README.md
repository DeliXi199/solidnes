# 0104 DeepQMC Spin-Penalty Alignment GPU Smoke

Purpose: validate the native FermiNet/PsiFormer `vmc_overlap` spin-penalty
implementation after changing it to DeepQMC-style loss-level
`beta * <S^2>` semantics.

## Fixed Settings

- Network: PsiFormer
- Attention: fused-QKV
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- `merge_keys: []`
- Diagonal MCMC, local energy, and overlap-JVP paths: enabled
- Spin penalty: `beta = 10.0` to match the public DeepQMC CLI example
- S2 diagnostics: enabled
- Batch size: 128
- Iterations: 2
- Optimizer: KFAC

## Configs

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_beta10_smoke_batch128_iter2.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_smoke_batch128_iter2.yaml
```

## Expected Outputs

```text
results/checkpoints/train_stats.csv
results/checkpoints/energy_matrix.npy
results/checkpoints/bare_energy_matrix.npy
results/checkpoints/s2_matrix.npy
results/checkpoints/overlap_matrix.npy
results/checkpoints/overlap_symmetric_matrix.npy
results/checkpoints/overlap_penalty_matrix.npy
results/checkpoints/overlap_gradient_scale.npy
results/checkpoints/overlap_state_ordering.npy
```

## Status

Completed through the approved FermiNet GPU submitter; no direct `sbatch`
command was used.

```text
job_id: 135738
partition/node: amdgpu40g/gpu004
state: COMPLETED
exit_code: 0:0
elapsed: 00:06:28
```

Validation:

```text
native_ferminet_excited_summary: ok
finite_outputs_check: ok
rows: 2
energy_matrix_frames: 2
bare_energy_matrix_frames: 2
s2_matrix_frames: 2
energy_matrix == bare_energy_matrix: true
final_energy_matrix: [-26.738569259643555, -27.948698043823242]
final_s2_matrix: [[3.2401905059814453, -0.009368719533085823], [-6.7874674797058105, 1.0729085206985474]]
```
