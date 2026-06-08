# 2026-06-08 Spin-Penalty And Default Learning-Rate Milestone

The excited-state spin-penalty implementation is complete for the current
SolidNES PsiFormer/FermiNet native route, and the default optimizer schedule
for future excited-state calculations has been selected from the 0113/0114
fresh-start eta/tau sweep.

## Milestone

Default excited-state calculations should use the DeepQMC-style loss-level spin
penalty path with state-specific local S2 targeting and the following optimizer
schedule unless a task explicitly declares a new sweep or ablation:

```text
spin_penalty / beta: 10.0
kfac.damping: 0.001
learning_rate / eta0: 0.02
learning_rate_delay / tau: 10000.0
learning_rate_decay: 1.0
```

The selected reference route is the validated FermiNet-QKV PsiFormer attention
configuration with two-state `vmc_overlap`, independent state parameters, and
`merge_keys: []`.

## Source Anchors

```text
src/solidnes/excited_state_mainline.py
src/solidnes/backends/ferminet_adapter.py
src/solidnes/excited_states/penalty.py
src/solidnes/excited_states/ferminet_pbc_adapter.py
scripts/validation/check_excited_state_mainline_defaults.py
scripts/validation/check_excited_state_penalty_objective.py
scripts/validation/check_ferminet_native_overlap_loss_alignment.py
scripts/validation/plot_psiformer_fixed_tau_eta_comparison.py
```

Default reference configs:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
```

## Spin-Penalty Completion

Implemented and validated behavior:

- spin penalty is applied as a loss-level `beta * <S^2>` term, not by folding
  spin into the Hamiltonian local energy;
- excited-state spin targeting uses a state-specific local S2 estimator aligned
  with DeepQMC `evaluate_spin`;
- native FermiNet/PsiFormer training records scalar spin diagnostics and
  optional per-state `spin_state_i` columns through `log_spin_by_state`;
- full S2 matrix observables are diagnostics and are disabled by default for
  production-scale spin-targeted runs;
- legacy fixed-ground paths remain explicit-only comparison branches;
- overlap penalty helper paths were aligned with the DeepQMC lower-state
  detached, centered-ratio tangent semantics.

## Parameter Evidence

The default schedule was selected from the combined 0113/0114 fresh-start
FermiNet-QKV sweep at fixed `beta=10`, `damping=0.001`, batch size 4096, and
30000 iterations.

Evidence source:

```text
tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/
```

Decision record:

```text
records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/default_parameter_decision.md
```

Final-window statistics use steps 29000-29999. Rolling plots use 1000-step
windows. The selected setting, `eta0=0.02` and `tau=10000`, gives near-best
final energies while keeping lower E0/E1/gap/spin fluctuation than the
`eta0=0.01, tau=10000` candidate and avoiding the larger spin/drift observed
for the `tau=15000` candidate.

Selected final-window statistics:

| Metric | Mean | Std | Var |
| --- | ---: | ---: | ---: |
| E0 Ha | -75.387869519 | 0.035995758 | 0.001295695 |
| E1 Ha | -75.182837834 | 0.036058101 | 0.001300187 |
| Gap eV | 5.579196363 | 1.380652527 | 1.906201399 |
| Spin `<S^2>` | 0.008607421 | 0.000666732 | 4.445320831e-7 |
| EWVar Ha^2 | 0.000654080 | 0.000342837 | n/a |

## Saved Data

Analysis snapshot:

```text
records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/fixed_tau_eta_comparison_analysis_20260608.tar.gz
```

SHA256:

```text
30e8d2bdb6795e67d2e156b17b2da9875f32ba7d2fee23d479cbbde5341aceca
```

The archive contains the fixed-tau analysis directory, the plotting script, and
the default train/experiment configs. Raw task `runs/`, checkpoints, and other
large generated runtime artifacts remain excluded from normal Git tracking.

## Validation

Passed before saving the milestone:

```text
python -m py_compile src/solidnes/excited_state_mainline.py scripts/validation/check_excited_state_mainline_defaults.py scripts/validation/plot_psiformer_fixed_tau_eta_comparison.py
.venv/ferminet-jax0101-cuda12/bin/python scripts/backends/build_ferminet_config.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
.venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_excited_state_mainline_defaults.py
PYTHONPATH=src .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_excited_state_penalty_objective.py
JAX_PLATFORMS=cpu .venv/ferminet-jax0101-cuda12/bin/python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
git diff --check
```

`check_ferminet_native_overlap_loss_alignment.py` emitted the expected local
FOLX warning about a potentially slow full-Hessian fallback and still completed
successfully.
