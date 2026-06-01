# 2026-06-01 Spin-Penalty DeepQMC Alignment

SolidNES now applies spin penalty in the native FermiNet/PsiFormer
`vmc_overlap` path with the same loss-level structure used by DeepQMC:

```text
L = weighted_energy + overlap_penalty + beta * <S^2>
```

## Implementation

Changed local external FermiNet patch files:

```text
external/ferminet/ferminet/loss.py
external/ferminet/ferminet/observables.py
external/ferminet/ferminet/train.py
patches/ferminet/deepqmc_spin_penalty_loss_level_2026-06-01.patch
```

Key behavior:

- `vmc_overlap` no longer folds `beta * S^2` into the Hamiltonian local energy.
- Hamiltonian local-energy clipping still acts on `H` only.
- The loss uses a state-specific local `S^2` estimator matching DeepQMC
  `evaluate_spin`, not the diagonal of FermiNet's full matrix S² observable.
- The spin custom-JVP coefficient is the per-state centered local `S^2`
  contribution with the final equal state average, matching the DeepQMC
  score-function structure.
- For DeepQMC-style `vmc_overlap` spin runs, `energy_matrix.npy` and
  `bare_energy_matrix.npy` are both Hamiltonian/bare diagnostics because the
  spin term is now loss-level.
- Non-`vmc_overlap` legacy objectives still use the older FermiNet effective
  local energy `H + beta S^2` path.

## Penalty Value

The initial DeepQMC-reference spin penalty value is kept as:

```text
spin_penalty: 10.0
```

This matches the public DeepQMC CLI example. It is a reference/starting value,
not a proven production value for periodic diamond.

## Validation

Passed:

```text
python -m compileall -q external/ferminet/ferminet/loss.py external/ferminet/ferminet/observables.py external/ferminet/ferminet/train.py scripts/validation/check_ferminet_native_overlap_loss_alignment.py
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_beta10_iter2000.yaml
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_state_specific_beta10_smoke_batch128_iter2.yaml
```

The validation script now includes synthetic checks for the DeepQMC-style spin
penalty forward value, rank-2 `[batch, states]` handling, custom-JVP gradient
coefficient, and state-specific local S² estimator.

JAX emitted the expected local CUDA plugin warning on this non-GPU shell, but
the validation completed successfully.

## SLURM GPU Smoke

Allocated and completed task `0105` through the approved wrapper after the
state-specific spin-estimator fix:

```text
task: tasks/excited_state_nesvmc/0105_deepqmc_spin_state_specific_gpu_smoke/
submitter: scripts/slurm/submit_ferminet_gpu_smoke.sh
job_id: 135764
partition/node: amdgpu40g/gpu004
state: COMPLETED
exit_code: 0:0
elapsed: 00:06:02
```

Smoke config:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_state_specific_beta10_smoke_batch128_iter2.yaml
```

Validation artifacts:

```text
results/validation/native_ferminet_excited_summary.json
results/validation/finite_outputs_check.log
```

Key checks passed:

- no-merge fused-QKV PsiFormer mainline route;
- `spin_penalty: 10.0` and `observables_s2: true`;
- state-specific DeepQMC-style local S² estimator is used by the loss;
- finite `train_stats.csv` rows;
- finite energy, bare-energy, overlap, overlap-scale, state-ordering, and
  `s2_matrix.npy` frames;
- `energy_matrix.npy == bare_energy_matrix.npy`, as expected for loss-level
  spin penalty diagnostics.

Earlier task `0104` also completed through the approved wrapper and validated
loss-level spin penalty before the final state-specific estimator correction:

```text
task: tasks/excited_state_nesvmc/0104_deepqmc_spin_penalty_alignment_gpu_smoke/
submitter: scripts/slurm/submit_ferminet_gpu_smoke.sh
job_id: 135738
partition/node: amdgpu40g/gpu004
state: COMPLETED
exit_code: 0:0
elapsed: 00:06:28
```

Smoke config:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_beta10_smoke_batch128_iter2.yaml
```

Validation artifacts:

```text
results/validation/native_ferminet_excited_summary.json
results/validation/finite_outputs_check.log
```

Key checks passed:

- no-merge fused-QKV PsiFormer mainline route;
- `spin_penalty: 10.0` and `observables_s2: true`;
- finite `train_stats.csv` rows;
- finite energy, bare-energy, overlap, overlap-scale, state-ordering, and
  `s2_matrix.npy` frames;
- `energy_matrix.npy == bare_energy_matrix.npy`, as expected for loss-level
  spin penalty diagnostics.
