"""Compatibility wrapper for upstream FermiNet PBC Hamiltonian."""

from __future__ import annotations

from typing import Any

from ferminet import hamiltonian
from ferminet import networks
from ferminet.pbc import hamiltonian as pbc_hamiltonian
import folx
import jax
import jax.numpy as jnp


def local_energy(*args: Any, ndim: int | None = None, **kwargs: Any):
    """Call upstream PBC local_energy while accepting train.py's ndim kwarg.

    Upstream FermiNet currently raises for ``states > 0`` in the PBC
    Hamiltonian even though the non-PBC Hamiltonian and loss stack support
    excited-state ``vmc_overlap``. SolidNES fills that gap here so native
    FermiNet excited-state training can use the same PBC Ewald potential and
    multi-state kinetic-energy machinery.
    """

    del ndim
    states = int(kwargs.get("states", 0) or 0)
    state_specific = bool(kwargs.get("state_specific", False))
    if states > 0 or state_specific:
        return _excited_pbc_local_energy(*args, **kwargs)
    return pbc_hamiltonian.local_energy(*args, **kwargs)


def _excited_pbc_local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = "default",
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = "ccecp",
    pp_symbols=None,
    lattice: jnp.ndarray | None = None,
    heg: bool = True,
    convergence_radius: int = 5,
) -> hamiltonian.LocalEnergy:
    """Create a PBC local-energy function for FermiNet native excited states."""

    del nspins
    del pp_type
    del use_scan
    if states <= 0:
        raise ValueError("states must be positive for excited-state PBC energy")
    if pp_symbols:
        raise NotImplementedError("Pseudopotentials are not implemented with PBC")
    if lattice is None:
        lattice = jnp.eye(3)

    matrix_kinetic = hamiltonian.excited_kinetic_energy_matrix(
        f,
        states,
        complex_output=complex_output,
        laplacian_method=laplacian_method,
    )

    def _pbc_potential_spectrum(data: networks.FermiNetData) -> jnp.ndarray:
        potential_energy = pbc_hamiltonian.make_ewald_potential(
            lattice,
            data.atoms,
            charges,
            convergence_radius,
            heg,
        )
        positions = jnp.reshape(data.positions, [states, -1])

        def one_state_potential(pos):
            ae, ee, _, _ = networks.construct_input_features(pos, data.atoms)
            return potential_energy(ae, ee)

        return jax.vmap(one_state_potential)(positions)

    def _state_specific_folx_kinetic(params, data):
        positions = jnp.reshape(data.positions, [states, -1])
        spins = jnp.reshape(data.spins, [states, -1])

        def one_state_kinetic(state_index: int):
            def state_log_psi(x):
                sign, logabs = f(params, x, spins[0], data.atoms, data.charges)
                return sign[state_index], logabs[state_index]

            sign_out, log_out = folx.forward_laplacian(
                state_log_psi,
                sparsity_threshold=6,
            )(positions[state_index])
            log_jac = log_out.jacobian.dense_array
            kinetic = -(log_out.laplacian + jnp.sum(log_jac * log_jac)) / 2
            if complex_output:
                sign_jac = sign_out.jacobian.dense_array
                kinetic -= 0.5j * sign_out.laplacian
                kinetic += 0.5 * jnp.sum(sign_jac * sign_jac)
                kinetic -= 1.0j * jnp.sum(sign_jac * log_jac)
            return kinetic

        return jnp.stack([one_state_kinetic(state) for state in range(states)])

    def _e_l(params, key, data: networks.FermiNetData):
        del key
        potential = _pbc_potential_spectrum(data)
        if state_specific:
            if laplacian_method != "folx":
                raise NotImplementedError(
                    "state_specific excited-state PBC energy currently requires folx"
                )
            return _state_specific_folx_kinetic(params, data) + potential, None

        psi_mat, kin_mat = matrix_kinetic(params, data)
        hpsi_mat = kin_mat + psi_mat * potential[:, None]
        energy_mat = jnp.linalg.solve(psi_mat, hpsi_mat)
        return jnp.trace(energy_mat), energy_mat

    return _e_l
