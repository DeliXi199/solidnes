"""Minimal FermiNet-PBC scaffold for two-state penalty objectives.

This module deliberately avoids importing FermiNet or JAX.  It defines the
array contract that a future FermiNet/JAX adapter can satisfy with vmapped
network and local-energy callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as _np

try:  # Keep this scaffold usable in admin environments without JAX installed.
    import jax.numpy as _jnp
except ModuleNotFoundError:  # pragma: no cover - exercised on machines without JAX.
    _jnp = _np

from solidnes.excited_states.overlap import estimate_overlap_from_ratios
from solidnes.excited_states.overlap import overlap_diagnostics
from solidnes.excited_states.penalty import penalty_vmc_terms


ArrayLike = Any
StateParams = Any
BatchedSignedNetwork = Callable[
    [StateParams, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    tuple[ArrayLike, ArrayLike],
]
BatchedLocalEnergy = Callable[
    [StateParams, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    ArrayLike,
]


@dataclass(frozen=True)
class FermiNetPBCStateSamples:
    """State-indexed FermiNet PBC walker data.

    All fields have leading axes `[states, walkers, ...]`.  A later JAX adapter
    can construct these arrays from one walker population per state.
    """

    positions: ArrayLike
    spins: ArrayLike
    atoms: ArrayLike
    charges: ArrayLike

    def __post_init__(self) -> None:
        nstates = self.nstates
        walkers = self.walkers
        for name, value in (
            ("spins", self.spins),
            ("atoms", self.atoms),
            ("charges", self.charges),
        ):
            shape = _jnp.asarray(value).shape
            if len(shape) < 2 or shape[0] != nstates or shape[1] != walkers:
                raise ValueError(
                    f"{name} must have leading axes [states, walkers] matching positions"
                )

    @property
    def nstates(self) -> int:
        shape = _jnp.asarray(self.positions).shape
        if len(shape) < 2:
            raise ValueError("positions must have leading axes [states, walkers]")
        return int(shape[0])

    @property
    def walkers(self) -> int:
        return int(_jnp.asarray(self.positions).shape[1])

    def for_sample_state(self, sample_state: int) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Return walker-batched FermiNet inputs for one state's samples."""

        return (
            _jnp.asarray(self.positions)[sample_state],
            _jnp.asarray(self.spins)[sample_state],
            _jnp.asarray(self.atoms)[sample_state],
            _jnp.asarray(self.charges)[sample_state],
        )


@dataclass(frozen=True)
class StateWavefunctionMatrix:
    """All state wavefunctions evaluated on all state walker populations."""

    sign: ArrayLike
    logabs: ArrayLike


@dataclass(frozen=True)
class StateEnergyEstimate:
    """State-wise local-energy samples and means."""

    local_energy: ArrayLike
    state_energy: ArrayLike


def broadcast_state_samples(
    *,
    positions: ArrayLike,
    spins: ArrayLike,
    atoms: ArrayLike,
    charges: ArrayLike,
) -> FermiNetPBCStateSamples:
    """Construct state samples from already state-indexed arrays.

    The explicit constructor documents intent at call sites and leaves room for
    future broadcasting helpers without changing downstream code.
    """

    return FermiNetPBCStateSamples(
        positions=_jnp.asarray(positions),
        spins=_jnp.asarray(spins),
        atoms=_jnp.asarray(atoms),
        charges=_jnp.asarray(charges),
    )


def evaluate_state_wavefunction_matrix(
    signed_network: BatchedSignedNetwork,
    state_params: Sequence[StateParams],
    samples: FermiNetPBCStateSamples,
) -> StateWavefunctionMatrix:
    """Evaluate every state wavefunction on every state's samples.

    `signed_network` must accept one state's params and walker-batched FermiNet
    inputs, returning `(sign, logabs)` with shape `[walkers]`.
    """

    nstates = len(state_params)
    if nstates != samples.nstates:
        raise ValueError("state_params length must match samples.nstates")

    sign_rows = []
    logabs_rows = []
    for params in state_params:
        sign_cols = []
        logabs_cols = []
        for sample_state in range(samples.nstates):
            positions, spins, atoms, charges = samples.for_sample_state(sample_state)
            sign, logabs = signed_network(params, positions, spins, atoms, charges)
            sign = _jnp.asarray(sign)
            logabs = _jnp.asarray(logabs)
            if sign.shape[0] != samples.walkers or logabs.shape[0] != samples.walkers:
                raise ValueError("signed_network outputs must have leading walker axis")
            sign_cols.append(sign)
            logabs_cols.append(logabs)
        sign_rows.append(_jnp.stack(sign_cols, axis=0))
        logabs_rows.append(_jnp.stack(logabs_cols, axis=0))
    return StateWavefunctionMatrix(
        sign=_jnp.stack(sign_rows, axis=0),
        logabs=_jnp.stack(logabs_rows, axis=0),
    )


def wavefunction_ratios_from_matrix(matrix: StateWavefunctionMatrix) -> ArrayLike:
    """Return `psi_i(r_j) / psi_j(r_j)` for all states and samples."""

    sign = _jnp.asarray(matrix.sign)
    logabs = _jnp.asarray(matrix.logabs)
    if sign.shape != logabs.shape or sign.ndim < 3:
        raise ValueError("sign and logabs must have shape [states, states, walkers]")
    if sign.shape[0] != sign.shape[1]:
        raise ValueError("state wavefunction matrix must be square on state axes")

    diag_sign = _jnp.diagonal(sign, axis1=0, axis2=1).T
    diag_logabs = _jnp.diagonal(logabs, axis1=0, axis2=1).T
    return (sign / diag_sign[None, :, :]) * _jnp.exp(logabs - diag_logabs[None, :, :])


def evaluate_overlap_diagnostics(
    signed_network: BatchedSignedNetwork,
    state_params: Sequence[StateParams],
    samples: FermiNetPBCStateSamples,
    *,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
) -> dict[str, ArrayLike]:
    """Evaluate overlap diagnostics from a batched FermiNet-like network."""

    matrix = evaluate_state_wavefunction_matrix(signed_network, state_params, samples)
    ratios = wavefunction_ratios_from_matrix(matrix)
    overlap = estimate_overlap_from_ratios(ratios, clip_upper=clip_upper)
    return overlap_diagnostics(overlap, collapse_threshold=collapse_threshold) | {
        "psi_ratio": ratios,
    }


def evaluate_state_energy_estimate(
    local_energy: BatchedLocalEnergy,
    state_params: Sequence[StateParams],
    samples: FermiNetPBCStateSamples,
) -> StateEnergyEstimate:
    """Evaluate each state's local energy on that state's own walkers only."""

    if len(state_params) != samples.nstates:
        raise ValueError("state_params length must match samples.nstates")

    per_state = []
    for state_idx, params in enumerate(state_params):
        positions, spins, atoms, charges = samples.for_sample_state(state_idx)
        values = _jnp.asarray(local_energy(params, positions, spins, atoms, charges))
        if values.shape[0] != samples.walkers:
            raise ValueError("local_energy output must have leading walker axis")
        per_state.append(values)
    local_energy_matrix = _jnp.stack(per_state, axis=0)
    return StateEnergyEstimate(
        local_energy=local_energy_matrix,
        state_energy=_jnp.mean(local_energy_matrix, axis=-1),
    )


def evaluate_penalty_scaffold_terms(
    signed_network: BatchedSignedNetwork,
    local_energy: BatchedLocalEnergy,
    state_params: Sequence[StateParams],
    samples: FermiNetPBCStateSamples,
    *,
    penalty_alpha: float,
    collapse_threshold: float = 0.95,
) -> dict[str, ArrayLike]:
    """Evaluate state energies, overlap diagnostics, and penalty objective terms."""

    energy = evaluate_state_energy_estimate(local_energy, state_params, samples)
    overlap = evaluate_overlap_diagnostics(
        signed_network,
        state_params,
        samples,
        collapse_threshold=collapse_threshold,
    )
    terms = penalty_vmc_terms(
        energy.state_energy,
        overlap["overlap_matrix"],
        penalty_alpha=penalty_alpha,
    )
    return {
        "local_energy": energy.local_energy,
        "state_energy": energy.state_energy,
        **overlap,
        **terms,
    }
