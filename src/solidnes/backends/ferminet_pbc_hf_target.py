"""PBC Hartree-Fock target evaluators for local FermiNet pretraining patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from solidnes.backends.ferminet_pbc_gto import PbcGtoEvaluator


@dataclass(frozen=True)
class JaxPbcGtoScfTarget:
    """Device-side PBC HF orbital target using the JAX PBC GTO MVP."""

    cell: Any
    kpts: np.ndarray
    coeff_alpha: jnp.ndarray
    coeff_beta: jnp.ndarray
    evaluator: PbcGtoEvaluator
    image_cutoff: int

    target_backend = "jax_pbc_gto"
    device_side_eval = True
    ndim = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_eval_mo",
            jax.jit(lambda coords, coeff: self.evaluator.eval_mo(coords, coeff)),
        )

    @classmethod
    def from_pbc_scf(
        cls,
        pbc_scf: Any,
        *,
        image_cutoff: int = 2,
    ) -> "JaxPbcGtoScfTarget":
        """Create a JAX target from an already-converged PySCF PBC HF wrapper."""
        kpts = np.asarray(pbc_scf.kpts)
        if kpts.shape != (1, 3):
            raise ValueError(
                "jax_pbc_gto target currently supports exactly one Gamma k-point; "
                f"got kpts shape {kpts.shape}"
            )
        if np.max(np.abs(kpts[0])) > 1e-12:
            raise ValueError("jax_pbc_gto target is currently Gamma-only.")
        coeff_alpha = np.asarray(pbc_scf.parameters["mo_coeff_alpha"])
        coeff_beta = np.asarray(pbc_scf.parameters["mo_coeff_beta"])
        evaluator = PbcGtoEvaluator.from_pyscf_cell(
            pbc_scf.cell,
            image_cutoff=image_cutoff,
            kpt=kpts[0],
        )
        return cls(
            cell=pbc_scf.cell,
            kpts=kpts,
            coeff_alpha=jnp.asarray(coeff_alpha),
            coeff_beta=jnp.asarray(coeff_beta),
            evaluator=evaluator,
            image_cutoff=int(image_cutoff),
        )

    def eval_orbitals(
        self,
        pos: jnp.ndarray,
        nspins: tuple[int, int],
        target_chunk_size: int = 0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate occupied PBC HF orbital matrices at electron positions."""
        del target_chunk_size
        pos = jnp.asarray(pos)
        leading_dims = pos.shape[:-1]
        nelec = int(sum(nspins))
        coord = pos.reshape(leading_dims + (nelec, self.ndim))
        return (
            self._eval_spin_orbitals(coord, nspins, spin=0),
            self._eval_spin_orbitals(coord, nspins, spin=1),
        )

    def _eval_spin_orbitals(
        self,
        coord: jnp.ndarray,
        nspins: tuple[int, int],
        *,
        spin: int,
    ) -> jnp.ndarray:
        ne = int(nspins[spin])
        if ne == 0:
            return jnp.zeros(coord.shape[:-2] + (0, 0), dtype=self.coeff_alpha.dtype)
        start = 0 if spin == 0 else int(nspins[0])
        stop = start + ne
        coeff = self.coeff_alpha if spin == 0 else self.coeff_beta
        spin_coord = coord[..., start:stop, :]
        flat_coord = spin_coord.reshape((-1, self.ndim))
        flat_mo = self._eval_mo(flat_coord, coeff)
        return flat_mo.reshape(coord.shape[:-2] + (ne, ne))
