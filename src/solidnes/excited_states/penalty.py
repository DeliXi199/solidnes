"""Penalty-objective utilities for excited-state VMC."""

from __future__ import annotations

from typing import Any

import numpy as _np

try:  # Keep these helpers importable outside the FermiNet/JAX environment.
    import jax.numpy as _jnp
except ModuleNotFoundError:  # pragma: no cover - exercised on machines without JAX.
    _jnp = _np

from solidnes.excited_states.overlap import offdiag_squared_overlap


ArrayLike = Any


def _asarray(value: ArrayLike) -> ArrayLike:
    return _jnp.asarray(value)


def weighted_state_energy(
    state_energies: ArrayLike,
    weights: ArrayLike | None = None,
) -> ArrayLike:
    """Return per-batch weighted state energy.

    The default matches the DeepQMC-style equal-state objective: average over
    the last axis, which is the state axis.
    """

    energies = _asarray(state_energies)
    if energies.ndim < 1:
        raise ValueError("state_energies must include a state axis")
    if weights is None:
        return _jnp.mean(energies, axis=-1)
    state_weights = _asarray(weights)
    if state_weights.shape != energies.shape[-1:]:
        raise ValueError("weights must have shape [states]")
    state_weights = state_weights / _jnp.sum(state_weights)
    return _jnp.sum(energies * state_weights, axis=-1)


def energy_gap_scale(
    energy_ewm: ArrayLike,
    *,
    min_gap_scale_factor: float = 0.1,
    max_scale_factor: float = 5.0,
) -> ArrayLike:
    """Scale overlap gradients by pairwise state-energy gaps."""

    energies = _asarray(energy_ewm)
    gap = _jnp.abs(energies[..., :, None] - energies[..., None, :])
    gap = _jnp.nan_to_num(gap, nan=1.0)
    return _jnp.clip(gap, min_gap_scale_factor, max_scale_factor)


def energy_std_scale(
    std_ewm: ArrayLike,
    *,
    min_gap_scale_factor: float = 0.01,
    max_scale_factor: float = 5.0,
) -> ArrayLike:
    """Scale overlap gradients by state energy standard deviations."""

    std = _jnp.nan_to_num(_asarray(std_ewm), nan=max_scale_factor)
    std = _jnp.clip(std, min_gap_scale_factor, max_scale_factor)
    return std[..., :, None]


def max_gap_std_scale(
    energy_ewm: ArrayLike,
    std_ewm: ArrayLike,
    *,
    min_gap_scale_factor: float = 0.1,
    max_scale_factor: float = 5.0,
) -> ArrayLike:
    """Scale overlap gradients by `max(pairwise energy gap, state std)`."""

    return _jnp.maximum(
        energy_gap_scale(
            energy_ewm,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
        ),
        energy_std_scale(
            std_ewm,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
        ),
    )


def penalty_vmc_terms(
    state_energies: ArrayLike,
    overlap_matrix: ArrayLike,
    *,
    penalty_alpha: float,
    energy_weights: ArrayLike | None = None,
) -> dict[str, ArrayLike]:
    """Return per-batch pieces of a penalty-based excited-state objective."""

    weighted_energy = weighted_state_energy(state_energies, energy_weights)
    overlap_penalty = offdiag_squared_overlap(overlap_matrix)
    total = weighted_energy + penalty_alpha * overlap_penalty
    return {
        "weighted_state_energy": weighted_energy,
        "offdiag_squared_overlap": overlap_penalty,
        "penalty_alpha": _jnp.asarray(penalty_alpha),
        "penalty_objective": total,
    }


def penalty_vmc_loss(
    state_energies: ArrayLike,
    overlap_matrix: ArrayLike,
    *,
    penalty_alpha: float,
    energy_weights: ArrayLike | None = None,
) -> ArrayLike:
    """Return the mean penalty-VMC objective over any leading batch axes."""

    return _jnp.mean(
        penalty_vmc_terms(
            state_energies,
            overlap_matrix,
            penalty_alpha=penalty_alpha,
            energy_weights=energy_weights,
        )["penalty_objective"]
    )
