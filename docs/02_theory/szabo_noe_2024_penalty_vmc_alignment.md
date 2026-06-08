# Szabo-Noe 2024 Penalty-VMC Alignment

Date: 2026-05-25

This document is the method contract for the SolidNES native FermiNet
implementation of the Szabo-Noe JCTC 2024 penalty-based excited-state VMC
method. Historical run notes live under `records/progress/`; this file is the
stable implementation spec.

## Primary Implementation Path

The primary implementation is the native FermiNet `vmc_overlap` objective:

```text
external/ferminet/ferminet/loss.py
external/ferminet/ferminet/train.py
src/solidnes/backends/ferminet_adapter.py
```

The older SolidNES external-state driver under `src/solidnes/excited_states/`
is retained as a diagnostic/reference path. New production method work should
target native FermiNet unless a specific adapter-level comparison is required.

## Method Profile

SolidNES exposes the paper-aligned defaults through:

```text
method_profile: szabo_noe_2024_penalty
```

The profile expands to:

```text
overlap_penalty: 4.0
overlap_scale_by: max_gap_std
overlap_min_scale: 0.001
overlap_max_scale: 5.0
overlap_clip_width: 10.0
overlap_clip_exclude_width: inf
overlap_sort_states_by: null
overlap_use_ewm_scale: true
overlap_ewm_max_alpha: 0.999
overlap_ewm_decay_alpha: 10.0
independent_state_params: true
kfac_norm_constraint_scale_by_states: false
```

## Objective

For state energies `E_i` and raw non-symmetric overlap estimates `x_ij`, the
forward loss is:

```text
L = sum_i w_i E_i + alpha * sum_{i < j} S_ij^2
S_ij^2 = max(x_ij * x_ji, 0)
S_ij = sign(x_ij) * sqrt(S_ij^2)
```

The DeepQMC-aligned profile uses equal state weights. In SolidNES configs,
`overlap_weights: null` under `method_profile: szabo_noe_2024_penalty` resolves
to a uniform tuple such as `(0.5, 0.5)` for two states. Explicit
`overlap_weights` values are still honored.

The scalar overlap penalty strength is the fixed DeepQMC/Szabo-Noe hyperparameter
`alpha = overlap_penalty = 4.0`. There is no separate automatic search for
`alpha` in the reference DeepQMC implementation. The automatic part of the
method is the overlap-gradient scaling in the custom tangent below.

With `independent_state_params: true`, native FermiNet/PsiFormer excited-state
runs initialize one complete network parameter tree per state. The multi-state
network API still returns `psi_i(x)` for every state, but parameters are not
shared across states; the states are coupled by the overlap-penalty objective.

For the fixed-ground experimental route, the DeepQMC-like choice is
`fixed_ground.symmetric_sampling: true`: estimate the fixed-reference penalty as
the product of an excited-sample ratio and an independent ground-sample ratio.
The older one-sided fixed-ground config is retained only as an explicit legacy
comparison and should not be treated as the aligned objective.

## Overlap Tangent

The custom-JVP overlap tangent follows the lower-state-detached upper-triangle
form used by the DeepQMC/Szabo-Noe reference implementation:

```text
psi ratios subtract each state's mean log amplitude before exponentiation
ratio gradients are median-MAD clipped
ratio gradients are centered over the sample axis
states use index ordering by default; energy ordering is an explicit control
only the ordered upper triangle contributes to the overlap tangent
the tangent is unpermuted back to the original state order
```

## DeepQMC-Style EWM Scaling

The native path carries DeepQMC-style running energy and
energy-standard-deviation estimates inside the FermiNet training batch.
`max_gap_std` scaling uses those running values:

```text
scale_ij = clip(max(abs(E_i^ewm - E_j^ewm), std_i^ewm), min_scale, max_scale)
```

The EWM is a finite-buffer weighted average with the same alpha schedule used by
DeepQMC:

```text
alpha_t = max(1 - max_alpha, 1 / (2 + t / decay_alpha))
weights_k = alpha_k * prod_{l<k}(1 - alpha_l)
```

Before the first EWM update, the loss sees NaN scale fields and applies the same
`nan_to_num` fallbacks as the scale functions. Subsequent steps use the EWM
values from previous training steps.

## Diagnostics

Native excited-state runs should write these diagnostics:

```text
energy_matrix.npy               # Hamiltonian local-energy diagnostics
bare_energy_matrix.npy          # optional mirror of H diagnostics for beta > 0 spin runs
overlap_matrix.npy              # raw non-symmetric x_ij
overlap_symmetric_matrix.npy    # signed S_ij
overlap_penalty_matrix.npy      # S_ij^2
overlap_gradient_scale.npy
overlap_state_ordering.npy
overlap_scale_energy_ewm.npy
overlap_scale_std_ewm.npy
```

Downstream summaries should not call the raw `x_ij` matrix an orthogonal overlap
without also reporting the symmetric and squared penalty matrices.

## Spin Penalty

Spin targeting is optional and separate from the default overlap method profile.
SolidNES maps the training key:

```text
spin_penalty: beta
```

to FermiNet's native:

```text
cfg.optim.spin_energy = beta
```

For DeepQMC-aligned native FermiNet/PsiFormer training, `beta > 0` is applied
as a loss-level term rather than by modifying the Hamiltonian local energy:

```text
L = energy_terms + overlap_terms + beta * <S^2>
```

For excited-state losses, the spin estimator is the state-specific local
estimator from DeepQMC `evaluate_spin`: for each sampled state walker, SolidNES
evaluates that state's wavefunction, swaps every up/down electron pair,
subtracts the signed-amplitude ratio from `S(S+1)+n_down`, and then averages
equally over states and samples. For single-state VMC-style losses, the same
loss-level machinery consumes the ordinary local `S^2` estimator. The custom JVP
uses DeepQMC's score-function form: the Hamiltonian energy coefficient is
clipped/centered as before, while the spin coefficient is the centered local
`S^2` contribution, with the final equal state average when states are present.
This avoids clipping `H + beta S^2` as one combined local energy.

SolidNES separates loss-level spin logging from FermiNet's full-matrix S2
observable:

```text
log_spin_by_state: true
observables_s2: true
```

`log_spin_by_state` writes `spin_state_0`, `spin_state_1`, ... to
`train_stats.csv` directly from the same DeepQMC-style spin estimator used by
the loss. `observables_s2` is separate and writes FermiNet's full matrix
observable `s2_matrix.npy`; keep it `false` for strict DeepQMC-style runs unless
the full matrix diagnostic is explicitly needed. DeepQMC-style spin-targeted
excited-state runs also write `bare_energy_matrix.npy`; because the spin term is
loss-level, this matrix is identical to the Hamiltonian `energy_matrix.npy` and
should be used as the physical gap diagnostic rather than the scalar training
objective.
The base
`szabo_noe_2024_penalty` profile keeps `spin_penalty: 0.0`; spin-targeted runs
should use an explicit spin config so the physical sector and penalty strength
are visible in the experiment YAML.  The initial DeepQMC-reference penalty
choice is `spin_penalty: 10.0`, matching the public DeepQMC CLI example.

For diamond Gamma with `(n_alpha, n_beta) = (6, 6)`, this targets the fixed
`S_z = 0` sector and penalizes spin contamination away from low `S^2`.

## KFAC Norm Constraint

For the DeepQMC-aligned `vmc_overlap` profile, SolidNES keeps
`cfg.optim.kfac.norm_constraint` at the YAML value and does not multiply it by
the number of states. This matches the explicit per-state parameterization,
where the state count already increases the trainable parameter tree.

## Explicit Non-Goals

The current spin penalty plumbing does not choose a universal penalty strength
or resolve singlet/triplet degeneracy by itself. Those remain experiment-level
choices that should be documented in the spin-targeted YAML.
