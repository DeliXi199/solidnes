# 2026-06-01 Spin-Penalty DeepQMC Alignment

SolidNES now applies spin penalty in all spin-enabled native
FermiNet/PsiFormer objectives with the same loss-level structure used by
DeepQMC:

```text
L = energy_terms + overlap_or_weighting_terms + beta * <S^2>
```

## Implementation

Changed implementation, validation, and patch files:

```text
external/ferminet/ferminet/loss.py
external/ferminet/ferminet/observables.py
external/ferminet/ferminet/train.py
scripts/validation/check_ferminet_native_overlap_loss_alignment.py
patches/ferminet/deepqmc_spin_penalty_loss_level_2026-06-01.patch
```

Key behavior:

- `vmc`, `wqmc`, `vmc_overlap`, and `fixed_ground_overlap` no longer fold
  `beta * S^2` into the Hamiltonian local energy.
- Hamiltonian local-energy clipping still acts on `H` only.
- Excited-state losses use a state-specific local `S^2` estimator matching
  DeepQMC `evaluate_spin`, not the diagonal of FermiNet's full matrix S²
  observable.
- Single-state VMC-style losses use the ordinary local `S^2` estimator through
  the same loss-level machinery.
- The spin custom-JVP coefficient is the centered local `S^2` contribution,
  with rank-1 `[batch]` and rank-2 `[batch, states]` cases handled explicitly
  and the final equal state average when states are present.
- 2026-06-02 follow-up: the custom JVP now reuses the loss-level
  `spin_tangent_coeff` computed from the same local `S^2` contributions instead
  of calling the expensive spin operator a second time in the tangent path.
  This matches the DeepQMC structure more closely and reduces redundant
  up/down exchange wavefunction evaluations for future runs.
- The same follow-up audited other custom-JVP overlap paths. Native
  `vmc_overlap` already reuses the forward `s_ij` ratio tensor in its tangent
  path. The fixed-ground overlap path, which is not used unless explicitly
  requested, now likewise carries a `fixed_ground_overlap_tangent_coeff` in aux
  data so its JVP no longer re-evaluates the fixed-reference overlap ratios.
- `train_stats.csv` records the observed `spin` and scalar `spin_penalty` when
  spin targeting is enabled; follow-up 0106 logging adds explicit
  `spin_state_i` CSV columns via `log_spin_by_state` without enabling the full
  S2 matrix observable.
- For DeepQMC-style spin-targeted excited-state runs, `energy_matrix.npy` and
  `bare_energy_matrix.npy` are both Hamiltonian/bare diagnostics because the
  spin term is loss-level.
- Follow-up audit corrected the beta=0 sweep config to stop expecting
  `bare_energy_matrix.npy`; with `spin_penalty: 0.0`, the Hamiltonian
  `energy_matrix.npy` and optional `s2_matrix.npy` diagnostics are sufficient.
- Follow-up orthogonality audit confirmed the native `vmc_overlap` penalty uses
  the DeepQMC clipped-geometric forward estimator and lower-state-detached
  overlap tangent. The fixed-ground adapter and native FermiNet train entry now
  default new fixed-ground configs to symmetric sampling; the older one-sided
  fixed-ground config is explicitly marked as a legacy comparison.
- Follow-up legacy-path audit aligned `src/solidnes/excited_states/` penalty
  helpers with the same semantics: forward `penalty_objective` uses the
  unscaled squared overlap, ratio clipping and overlap-gradient scale apply
  only to the custom tangent/surrogate, and the external-state FermiNet PBC
  surrogate now uses the DeepQMC lower-state-detached centered-ratio form
  rather than direct differentiation of the overlap product.

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
python -m compileall -q src/solidnes/backends/ferminet_adapter.py external/ferminet/ferminet/loss.py external/ferminet/ferminet/observables.py external/ferminet/ferminet/train.py scripts/validation/check_ferminet_native_overlap_loss_alignment.py
python -m compileall -q src/solidnes/excited_states/penalty.py src/solidnes/excited_states/ferminet_pbc_adapter.py src/solidnes/excited_states/ferminet_pbc_scaffold.py scripts/validation/check_excited_state_penalty_objective.py scripts/validation/check_ferminet_pbc_penalty_grad_step.py
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
JAX_PLATFORMS=cpu PYTHONPATH=src:external/ferminet .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_ferminet_pbc_penalty_grad_step.py --states 2 --walkers 2 --platform cpu
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_beta10_iter2000.yaml
SOLIDNES_BUILD_ONLY=1 JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_spin_state_specific_beta10_smoke_batch128_iter2.yaml
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_fixed_ground_overlap_beta0_iter20000.yaml --build-only
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_ferminet_pbc_gamma_fixed_ground_symmetric_overlap_beta0_iter20000.yaml --build-only
```

The validation script now includes synthetic checks for the DeepQMC-style spin
penalty forward value, rank-1 `[batch]` and rank-2 `[batch, states]` handling,
VMC/WQMC/overlap/fixed-ground custom-JVP coefficients, and the state-specific
local S² estimator.

It also now checks the overlap-specific tangent more tightly: a three-state
ordered/scaled case verifies the DeepQMC lower-state-detached coefficient, and a
fixed-ground symmetric custom-JVP toy problem verifies the two-sided
fixed-reference penalty path.

JAX/FOLX emitted expected local warnings on this shell, but the validation
completed successfully.

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
