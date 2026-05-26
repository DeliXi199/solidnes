#!/usr/bin/env python3
"""No-compute synthetic checks for excited-state penalty utilities."""

from __future__ import annotations

import math

import numpy as np

from solidnes.excited_states import energy_gap_scale
from solidnes.excited_states import energy_std_scale
from solidnes.excited_states import estimate_overlap_from_ratios
from solidnes.excited_states import max_gap_std_scale
from solidnes.excited_states import overlap_gradient_scale
from solidnes.excited_states import overlap_diagnostics
from solidnes.excited_states import overlap_penalty_loss
from solidnes.excited_states import penalty_vmc_loss
from solidnes.excited_states import penalty_vmc_terms
from solidnes.excited_states import scaled_offdiag_squared_overlap
from solidnes.excited_states import clip_psi_ratios_by_median
from solidnes.excited_states import symmetrize_overlap_with_clipped_geometric_mean


def _assert_close(actual, expected, *, tol=1e-12):
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=tol, rtol=tol):
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


def main() -> None:
    non_symmetric = np.array([[1.0, 0.25], [0.16, 1.0]])
    symmetric = symmetrize_overlap_with_clipped_geometric_mean(non_symmetric)
    _assert_close(symmetric, np.array([[1.0, 0.2], [0.2, 1.0]]))

    signed = np.array([[1.0, -0.25], [0.16, 1.0]])
    signed_symmetric = symmetrize_overlap_with_clipped_geometric_mean(signed)
    _assert_close(signed_symmetric, np.array([[1.0, 0.0], [0.0, 1.0]]))

    ratios = np.array(
        [
            [[1.0, 1.0], [0.2, 0.3]],
            [[0.1, 0.22], [1.0, 1.0]],
        ]
    )
    clipped_ratios, ratio_mask = clip_psi_ratios_by_median(ratios, clip_width=1.0)
    _assert_close(clipped_ratios[0, 1], [0.2, 0.3])
    _assert_close(ratio_mask, np.ones_like(ratios, dtype=bool))

    overlap = estimate_overlap_from_ratios(ratios)
    expected_overlap_01 = math.sqrt(0.25 * 0.16)
    _assert_close(overlap, np.array([[1.0, expected_overlap_01], [expected_overlap_01, 1.0]]))
    _assert_close(overlap_penalty_loss(overlap), expected_overlap_01**2)

    diagnostics = overlap_diagnostics(overlap, collapse_threshold=0.19)
    _assert_close(diagnostics["max_abs_offdiag_overlap"], expected_overlap_01)
    if bool(np.asarray(diagnostics["collapse_flag"])) is not True:
        raise AssertionError("collapse flag should trigger at threshold 0.19")

    state_energies = np.array([-75.0, -74.8])
    terms = penalty_vmc_terms(state_energies, overlap, penalty_alpha=4.0)
    expected_weighted_energy = -74.9
    expected_loss = expected_weighted_energy + 4.0 * expected_overlap_01**2
    _assert_close(terms["weighted_state_energy"], expected_weighted_energy)
    _assert_close(penalty_vmc_loss(state_energies, overlap, penalty_alpha=4.0), expected_loss)

    scale = np.array([[0.1, 0.2], [0.3, 0.4]])
    _assert_close(scaled_offdiag_squared_overlap(overlap, scale), 0.2 * expected_overlap_01**2)
    scaled_terms = penalty_vmc_terms(
        state_energies,
        overlap,
        penalty_alpha=4.0,
        overlap_scale=scale,
    )
    _assert_close(
        scaled_terms["penalty_objective"],
        expected_weighted_energy + 4.0 * 0.2 * expected_overlap_01**2,
    )

    energies = np.array([-75.0, -74.8])
    std = np.array([0.05, 0.3])
    _assert_close(energy_gap_scale(energies, min_gap_scale_factor=0.1), [[0.1, 0.2], [0.2, 0.1]])
    _assert_close(energy_std_scale(std, min_gap_scale_factor=0.1), [[0.1], [0.3]])
    _assert_close(max_gap_std_scale(energies, std, min_gap_scale_factor=0.1), [[0.1, 0.2], [0.3, 0.3]])
    _assert_close(
        overlap_gradient_scale(
            energies,
            std,
            scale_by="max_gap_std",
            min_gap_scale_factor=0.1,
        ),
        [[0.1, 0.2], [0.3, 0.3]],
    )

    print("excited-state penalty objective synthetic checks passed")


if __name__ == "__main__":
    main()
