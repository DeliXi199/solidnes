"""Overlap estimators and diagnostics for penalty-based excited-state VMC."""

from __future__ import annotations

from typing import Any

import numpy as _np

try:  # Keep these helpers importable in lightweight CPU/admin environments.
    import jax.numpy as _jnp
except ModuleNotFoundError:  # pragma: no cover - exercised on machines without JAX.
    _jnp = _np


ArrayLike = Any


def _asarray(value: ArrayLike) -> ArrayLike:
    return _jnp.asarray(value)


def _triu_values(matrix: ArrayLike, k: int = 1) -> ArrayLike:
    matrix = _asarray(matrix)
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("overlap matrix must be square on its last two axes")
    row_idx, col_idx = _np.triu_indices(matrix.shape[-1], k=k)
    return matrix[..., row_idx, col_idx]


def symmetrize_overlap_with_clipped_geometric_mean(
    non_symmetric_overlap: ArrayLike,
    *,
    clip_upper: bool = False,
) -> ArrayLike:
    """Symmetrize non-symmetric overlap estimates from state-wise samples.

    `non_symmetric_overlap[..., i, j]` is the estimate of
    `E_{r ~ |psi_j|^2}[psi_i(r) / psi_j(r)]`.  The default lower-bound clipping
    follows the inspected DeepQMC implementation.  `clip_upper=True` also caps
    the geometric product at one, matching the DeepQMC docstring convention.
    """

    x = _asarray(non_symmetric_overlap)
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("non_symmetric_overlap must be square on its last two axes")
    product = x * _jnp.swapaxes(x, -1, -2)
    product = _jnp.clip(product, 0.0, 1.0) if clip_upper else _jnp.maximum(product, 0.0)
    return _jnp.sign(x) * _jnp.sqrt(product)


def estimate_overlap_from_ratios(
    psi_ratio: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    symmetrize: bool = True,
    clip_upper: bool = False,
) -> ArrayLike:
    """Estimate the overlap matrix from all-state wavefunction ratios.

    Args:
        psi_ratio: Array with shape `[..., states, states, walkers]`, where
            `psi_ratio[..., i, j, k] = psi_i(r_jk) / psi_j(r_jk)` and samples
            `r_jk` are drawn from `|psi_j|^2`.
        weights: Optional sample weights with shape `[..., states, walkers]`.
            If omitted, an unweighted average over walkers is used.
        symmetrize: Whether to apply clipped-geometric-mean symmetrization.
        clip_upper: Whether to cap the symmetrization product at one.
    """

    ratios = _asarray(psi_ratio)
    if ratios.ndim < 3:
        raise ValueError("psi_ratio must have shape [..., states, states, walkers]")
    if ratios.shape[-2] != ratios.shape[-3]:
        raise ValueError("psi_ratio state axes must have the same length")

    if weights is None:
        overlap = _jnp.mean(ratios, axis=-1)
    else:
        sample_weights = _asarray(weights)
        expected_shape = ratios.shape[:-3] + ratios.shape[-2:]
        if sample_weights.shape != expected_shape:
            raise ValueError(
                "weights must have shape [..., states, walkers] matching psi_ratio"
            )
        weighted_sum = _jnp.sum(ratios * sample_weights[..., None, :, :], axis=-1)
        norm = _jnp.sum(sample_weights, axis=-1)
        overlap = weighted_sum / norm[..., None, :]

    if not symmetrize:
        return overlap
    return symmetrize_overlap_with_clipped_geometric_mean(overlap, clip_upper=clip_upper)


def clip_psi_ratios_by_median(
    psi_ratio: ArrayLike,
    *,
    clip_width: float = 10.0,
    exclude_width: float = _np.inf,
) -> tuple[ArrayLike, ArrayLike]:
    """Clip wavefunction ratios along the walker axis and return a grad mask."""

    ratios = _asarray(psi_ratio)
    center = _jnp.median(ratios, axis=-1, keepdims=True)
    deviation = _jnp.abs(ratios - center)
    sigma = _jnp.median(deviation, axis=-1, keepdims=True)
    lower = center - clip_width * sigma
    upper = center + clip_width * sigma
    clipped = _jnp.clip(ratios, lower, upper)
    mask = deviation < exclude_width
    return clipped, mask


def offdiag_squared_overlap(overlap_matrix: ArrayLike) -> ArrayLike:
    """Return per-batch `sum_{i < j} S_ij^2`."""

    offdiag = _triu_values(overlap_matrix, k=1)
    if offdiag.shape[-1] == 0:
        return _jnp.zeros(_asarray(overlap_matrix).shape[:-2])
    return _jnp.sum(offdiag * offdiag, axis=-1)


def scaled_offdiag_squared_overlap(
    overlap_matrix: ArrayLike,
    overlap_scale: ArrayLike,
) -> ArrayLike:
    """Return `sum_{i < j} scale_ij * S_ij^2`."""

    offdiag = _triu_values(overlap_matrix, k=1)
    if offdiag.shape[-1] == 0:
        return _jnp.zeros(_asarray(overlap_matrix).shape[:-2])
    scale = _asarray(overlap_scale)
    if scale.ndim == 0:
        offdiag_scale = scale
    else:
        offdiag_scale = _triu_values(scale, k=1)
    return _jnp.sum(offdiag_scale * offdiag * offdiag, axis=-1)


def overlap_penalty_loss(overlap_matrix: ArrayLike) -> ArrayLike:
    """Return the mean squared upper-triangle overlap penalty."""

    return _jnp.mean(offdiag_squared_overlap(overlap_matrix))


def overlap_diagnostics(
    overlap_matrix: ArrayLike,
    *,
    collapse_threshold: float = 0.95,
) -> dict[str, ArrayLike]:
    """Compute lightweight diagnostics for state orthogonality/collapse."""

    overlap = _asarray(overlap_matrix)
    offdiag = _jnp.abs(_triu_values(overlap, k=1))
    if offdiag.shape[-1] == 0:
        max_abs_offdiag = _jnp.zeros(overlap.shape[:-2])
    else:
        max_abs_offdiag = _jnp.max(offdiag, axis=-1)
    per_batch_penalty = offdiag_squared_overlap(overlap)
    return {
        "overlap_matrix": overlap,
        "offdiag_squared_overlap": per_batch_penalty,
        "overlap_penalty": _jnp.mean(per_batch_penalty),
        "max_abs_offdiag_overlap": max_abs_offdiag,
        "collapse_flag": max_abs_offdiag >= collapse_threshold,
    }
