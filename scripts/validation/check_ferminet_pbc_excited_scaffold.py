#!/usr/bin/env python3
"""No-compute synthetic check for the FermiNet PBC excited-state scaffold."""

from __future__ import annotations

import numpy as np

from solidnes.excited_states.ferminet_pbc_scaffold import broadcast_state_samples
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_overlap_diagnostics
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_penalty_scaffold_terms
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_wavefunction_matrix
from solidnes.excited_states.ferminet_pbc_scaffold import wavefunction_ratios_from_matrix


def _fake_signed_network(params, positions, spins, atoms, charges):
    del spins, atoms, charges
    x = np.asarray(positions)[:, 0]
    center = float(params["center"])
    logabs = -0.5 * (x - center) ** 2
    sign = np.ones_like(logabs)
    return sign, logabs


def _fake_local_energy(params, positions, spins, atoms, charges):
    del spins, atoms, charges
    x = np.asarray(positions)[:, 0]
    return float(params["energy"]) + 0.01 * x


def _assert_close(actual, expected, *, tol=1e-12):
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=tol, rtol=tol):
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


def main() -> None:
    positions = np.array(
        [
            [[-0.1], [0.0], [0.1]],
            [[0.9], [1.0], [1.1]],
        ]
    )
    zeros = np.zeros_like(positions)
    samples = broadcast_state_samples(
        positions=positions,
        spins=zeros,
        atoms=zeros,
        charges=zeros,
    )
    params = [
        {"center": 0.0, "energy": -75.0},
        {"center": 1.0, "energy": -74.8},
    ]

    matrix = evaluate_state_wavefunction_matrix(_fake_signed_network, params, samples)
    if np.asarray(matrix.sign).shape != (2, 2, 3):
        raise AssertionError("unexpected state wavefunction matrix shape")

    ratios = wavefunction_ratios_from_matrix(matrix)
    _assert_close(ratios[0, 0], np.ones(3))
    _assert_close(ratios[1, 1], np.ones(3))

    diagnostics = evaluate_overlap_diagnostics(
        _fake_signed_network,
        params,
        samples,
        collapse_threshold=0.95,
    )
    overlap = np.asarray(diagnostics["overlap_matrix"])
    if overlap.shape != (2, 2):
        raise AssertionError("unexpected overlap matrix shape")
    if not (0.0 <= overlap[0, 1] < 1.0):
        raise AssertionError(f"unexpected off-diagonal overlap: {overlap[0, 1]}")

    terms = evaluate_penalty_scaffold_terms(
        _fake_signed_network,
        _fake_local_energy,
        params,
        samples,
        penalty_alpha=4.0,
    )
    _assert_close(terms["state_energy"], np.array([-75.0, -74.79]))
    expected_objective = np.mean(terms["state_energy"]) + 4.0 * terms["offdiag_squared_overlap"]
    _assert_close(terms["penalty_objective"], expected_objective)

    print("FermiNet PBC excited-state scaffold synthetic checks passed")


if __name__ == "__main__":
    main()
