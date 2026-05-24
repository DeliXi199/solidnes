"""Minimal JAX periodic GTO evaluator for FermiNet PBC pretraining tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ferminet.utils import gto as ferminet_gto
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto as molecular_gto


@dataclass(frozen=True)
class PbcGtoEvaluator:
    """Evaluate Gamma/k-point periodic AOs by finite lattice image sums.

    This is intentionally small: it is an MVP for validating AO/MO ordering and
    normalization against PySCF PBC before wiring the evaluator into pretraining.
    """

    mol: ferminet_gto.Mol
    lattice: jnp.ndarray
    translations: jnp.ndarray
    kpt: jnp.ndarray

    @classmethod
    def from_pyscf_cell(
        cls,
        cell: Any,
        *,
        image_cutoff: int = 1,
        kpt: np.ndarray | None = None,
    ) -> "PbcGtoEvaluator":
        """Build from a PySCF PBC cell.

        Args:
            cell: Built PySCF ``pbc.gto.Cell``.
            image_cutoff: Integer image radius in lattice-vector units.
            kpt: Cartesian k-point in PySCF convention. Defaults to Gamma.
        """
        if image_cutoff < 0:
            raise ValueError(f"image_cutoff must be non-negative, got {image_cutoff}")
        if not getattr(cell, "_built", False):
            cell = cell.copy()
            cell.build()

        mol = molecular_gto.Mole()
        mol.atom = cell.atom
        mol.unit = getattr(cell, "unit", "B") or "B"
        mol.basis = cell.basis
        mol.charge = getattr(cell, "charge", 0)
        mol.spin = getattr(cell, "spin", 0)
        mol.verbose = 0
        mol.build()

        lattice = np.asarray(cell.a, dtype=np.float64)
        translations = make_lattice_translations(lattice, image_cutoff)
        if kpt is None:
            kpt = np.zeros(3, dtype=np.float64)
        return cls(
            mol=ferminet_gto.Mol.from_pyscf_mol(mol),
            lattice=jnp.asarray(lattice),
            translations=jnp.asarray(translations),
            kpt=jnp.asarray(kpt, dtype=jnp.float64),
        )

    def eval_ao(self, coords: jnp.ndarray, *, wrap: bool = True) -> jnp.ndarray:
        """Evaluate periodic AO values at ``coords``.

        Args:
            coords: Cartesian coordinates with trailing dimension 3.
            wrap: If true, first wrap coordinates into the PySCF cell and apply
                the corresponding k-point phase. Gamma-point results are
                unchanged by this phase.

        Returns:
            AO array with shape ``coords.shape[:-1] + (nao,)``.
        """
        coords = jnp.asarray(coords)
        leading_shape = coords.shape[:-1]
        flat = coords.reshape(-1, 3)
        if wrap:
            flat, cell_shift = enforce_pbc(self.lattice, flat)
            coord_phase = jnp.exp(1j * jnp.einsum("d,nd->n", self.kpt, cell_shift))
        else:
            coord_phase = jnp.ones(flat.shape[0], dtype=jnp.complex128)

        translation_phase = jnp.exp(
            1j * jnp.einsum("d,td->t", self.kpt, self.translations)
        )

        def shifted_ao(translation, phase):
            return self.mol.eval_gto(flat - translation) * phase

        ao = jnp.sum(
            jax.vmap(shifted_ao)(self.translations, translation_phase), axis=0
        )
        ao = ao * jnp.asarray(coord_phase)[:, None]
        return ao.reshape(leading_shape + (ao.shape[-1],))

    def eval_mo(
        self,
        coords: jnp.ndarray,
        mo_coeff: jnp.ndarray,
        *,
        wrap: bool = True,
    ) -> jnp.ndarray:
        """Project periodic AOs into molecular orbitals."""
        return self.eval_ao(coords, wrap=wrap) @ jnp.asarray(mo_coeff)


def make_lattice_translations(lattice: np.ndarray, image_cutoff: int) -> np.ndarray:
    """Return lattice translations for a cubic image shell."""
    image_range = np.arange(-image_cutoff, image_cutoff + 1, dtype=np.int64)
    mesh = np.meshgrid(image_range, image_range, image_range, indexing="ij")
    indices = np.stack(mesh, axis=-1).reshape(-1, 3)
    norms = np.linalg.norm(indices, axis=1)
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0], norms))
    return indices[order] @ np.asarray(lattice, dtype=np.float64)


def enforce_pbc(lattice: jnp.ndarray, coords: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wrap Cartesian coordinates into the cell and return cell translations."""
    fractional = coords @ jnp.linalg.inv(lattice)
    cell_shift = jnp.floor(fractional)
    wrapped = (fractional - cell_shift) @ lattice
    return wrapped, cell_shift @ lattice
