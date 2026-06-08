"""Penalty-objective utilities for excited-state VMC."""

from __future__ import annotations

from typing import Any

import numpy as _np

try:  # Keep these helpers importable outside the FermiNet/JAX environment.
    import jax.numpy as _jnp
except ModuleNotFoundError:  # pragma: no cover - exercised on machines without JAX.
    _jnp = _np

from solidnes.excited_states.overlap import offdiag_squared_overlap
from solidnes.excited_states.overlap import scaled_offdiag_squared_overlap


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


def overlap_gradient_scale(
    state_energies: ArrayLike,
    state_energy_std: ArrayLike | None = None,
    *,
    scale_by: str | None = "max_gap_std",
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
) -> ArrayLike:
    """Return pairwise overlap-gradient scaling factors."""

    energies = _asarray(state_energies)
    if scale_by is None or scale_by == "none":
        return _jnp.ones(energies.shape[:-1] + (energies.shape[-1], energies.shape[-1]))
    if scale_by == "energy_gap":
        return energy_gap_scale(
            energies,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
        )
    if state_energy_std is None:
        std = _jnp.zeros_like(energies)
    else:
        std = _asarray(state_energy_std)
    if scale_by == "energy_std":
        std_scale = energy_std_scale(
            std,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
        )
        return _jnp.broadcast_to(std_scale, energies.shape + energies.shape[-1:])
    if scale_by == "max_gap_std":
        return max_gap_std_scale(
            energies,
            std,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
        )
    raise ValueError(f"Unsupported overlap scale_by: {scale_by}")


def penalty_vmc_terms(
    state_energies: ArrayLike,
    overlap_matrix: ArrayLike,
    *,
    penalty_alpha: float,
    energy_weights: ArrayLike | None = None,
    overlap_scale: ArrayLike | None = None,
) -> dict[str, ArrayLike]:
    """Return per-batch pieces of a penalty-based excited-state objective.

    DeepQMC keeps the forward overlap objective unscaled; ``overlap_scale`` is
    a gradient/tangent preconditioner.  We still report the scaled diagnostic so
    legacy external-state training can use it for its paper-style surrogate.
    """

    weighted_energy = weighted_state_energy(state_energies, energy_weights)
    overlap_penalty = offdiag_squared_overlap(overlap_matrix)
    if overlap_scale is None:
        scaled_overlap_penalty = overlap_penalty
        scale = _jnp.asarray(1.0)
    else:
        scale = _asarray(overlap_scale)
        scaled_overlap_penalty = scaled_offdiag_squared_overlap(overlap_matrix, scale)
    total = weighted_energy + penalty_alpha * overlap_penalty
    return {
        "weighted_state_energy": weighted_energy,
        "offdiag_squared_overlap": overlap_penalty,
        "scaled_offdiag_squared_overlap": scaled_overlap_penalty,
        "overlap_gradient_scale": scale,
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
