# 0105 DeepQMC State-Specific Spin GPU Smoke

Purpose: validate the final DeepQMC-aligned `vmc_overlap` spin penalty after
switching the loss operator to DeepQMC's state-specific local S² estimator.

## Fixed Settings

- Network: PsiFormer
- Attention: fused-QKV
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- `merge_keys: []`
- Diagonal MCMC, local energy, and overlap-JVP paths: enabled
- Spin penalty: `beta = 10.0` to match the public DeepQMC CLI example
- Spin estimator used in loss: state-specific DeepQMC `evaluate_spin` analogue
- S² matrix diagnostics: enabled separately as observable
- Batch size: 128
- Iterations: 2
- Optimizer: KFAC

## Config

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_state_specific_beta10_smoke_batch128_iter2.yaml
```

## Status

Completed through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.

```text
job_id: 135764
partition/node: amdgpu40g/gpu004
state: COMPLETED
exit_code: 0:0
elapsed: 00:06:02
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
final_energy_matrix: [-26.563156127929688, -27.983013153076172]
final_s2_matrix: [[1.5180895328521729, -0.03673870116472244], [1.2818844318389893, 1.6371718645095825]]
```
