"""Paper-geometry Gamma-point all-electron diamond config for FermiNet.

The geometry follows the DeepSolid supplementary benchmark table for the
1x1x1 diamond primitive cell: primitive lattice vector components are 1.7869 A,
equivalent to a conventional cubic lattice constant of 3.5738 A.
"""

from __future__ import annotations

import numpy as np

from ferminet import base_config
from ferminet.pbc import envelopes
from ferminet.utils import system


BOHR_PER_ANGSTROM = 1.889726124565062
DIAMOND_LATTICE_CONSTANT_ANGSTROM = 3.5738


def diamond_primitive_lattice(
    lattice_constant_angstrom: float = DIAMOND_LATTICE_CONSTANT_ANGSTROM,
) -> np.ndarray:
    """Return diamond primitive lattice vectors in bohr as column vectors."""

    lattice_constant_bohr = lattice_constant_angstrom * BOHR_PER_ANGSTROM
    return 0.5 * lattice_constant_bohr * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def diamond_primitive_atoms(lattice: np.ndarray) -> list[system.Atom]:
    """Return the two carbon atoms in a primitive diamond cell."""

    fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    cartesian_bohr = fractional @ lattice.T
    return [
        system.Atom("C", tuple(cartesian_bohr[0])),
        system.Atom("C", tuple(cartesian_bohr[1])),
    ]


def get_config():
    """Build a FermiNet config for the paper-geometry diamond benchmark."""

    cfg = base_config.default()
    lattice = diamond_primitive_lattice()

    cfg.system.molecule = diamond_primitive_atoms(lattice)
    cfg.system.electrons = (6, 6)
    cfg.system.units = "bohr"
    cfg.system.make_local_energy_fn = (
        "solidnes.backends.ferminet_pbc_hamiltonian.local_energy"
    )
    cfg.system.make_local_energy_kwargs = {
        "lattice": lattice,
        "heg": False,
        "convergence_radius": 5,
    }

    cfg.network.make_feature_layer_fn = (
        "ferminet.pbc.feature_layer.make_pbc_feature_layer"
    )
    cfg.network.make_feature_layer_kwargs = {
        "lattice": lattice,
        "include_r_ae": True,
    }

    kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)
    cfg.network.make_envelope_fn = "ferminet.pbc.envelopes.make_multiwave_envelope"
    cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
    cfg.network.full_det = True

    # Upstream FermiNet pretraining targets molecular PySCF orbitals. It does
    # not currently provide a periodic HF pretraining target for this PBC path.
    cfg.pretrain.method = None
    cfg.pretrain.iterations = 0
    return cfg
