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
overlap_sort_states_by: energy
overlap_use_ewm_scale: true
overlap_ewm_max_alpha: 0.999
overlap_ewm_decay_alpha: 10.0
kfac_norm_constraint_scale_by_states: true
```

## Objective

For state energies `E_i` and raw non-symmetric overlap estimates `x_ij`, the
forward loss is:

```text
L = sum_i w_i E_i + alpha * sum_{i < j} S_ij^2
S_ij^2 = max(x_ij * x_ji, 0)
S_ij = sign(x_ij) * sqrt(S_ij^2)
```

The default FermiNet state weights are retained unless explicitly overridden by
`overlap_weights`.

The scalar overlap penalty strength is the fixed DeepQMC/Szabo-Noe hyperparameter
`alpha = overlap_penalty = 4.0`. There is no separate automatic search for
`alpha` in the reference DeepQMC implementation. The automatic part of the
method is the overlap-gradient scaling in the custom tangent below.

## Overlap Tangent

The custom-JVP overlap tangent follows the lower-state-detached upper-triangle
form used by the DeepQMC/Szabo-Noe reference implementation:

```text
ratio gradients are median-MAD clipped
ratio gradients are centered over the sample axis
states are optionally ordered by running energy
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
energy_matrix.npy               # training local energies; spin-penalized if beta > 0
bare_energy_matrix.npy          # optional, H-only energies for spin runs
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

When `beta > 0`, the native FermiNet path minimizes the effective local energy
`H + beta S^2`. For state-specific `vmc_overlap` local energies, SolidNES adds
`beta * diag(S^2)` to the per-state local-energy vector. For matrix-energy
paths, it adds `beta * trace(S^2)` to the scalar objective and `beta * S^2` to
the auxiliary local-energy matrix. This follows the DeepQMC/Szabo-Noe optional
penalty form:

```text
L += spin_penalty * <S^2>
```

SolidNES also exposes:

```text
observables_s2: true
```

which writes `s2_matrix.npy` for native excited-state runs. If
`observables_s2` is not set, SolidNES enables it automatically when
`spin_penalty > 0`. Spin-targeted runs also write `bare_energy_matrix.npy`,
which removes the spin penalty from the training energy diagnostics and should
be used for physical excitation gaps. The base `szabo_noe_2024_penalty` profile
keeps `spin_penalty: 0.0`; spin-targeted runs should use an explicit spin
config so the physical sector and penalty strength are visible in the
experiment YAML.

For diamond Gamma with `(n_alpha, n_beta) = (6, 6)`, this targets the fixed
`S_z = 0` sector and penalizes spin contamination away from low `S^2`.

## KFAC Norm Constraint

For multi-state native FermiNet `vmc_overlap`, the method profile scales
`cfg.optim.kfac.norm_constraint` by the number of states. The YAML value remains
the single-state base value; the adapter writes the effective value into the
FermiNet config and build summary.

## Explicit Non-Goals

The current spin penalty plumbing does not choose a universal penalty strength
or resolve singlet/triplet degeneracy by itself. Those remain experiment-level
choices that should be documented in the spin-targeted YAML.
