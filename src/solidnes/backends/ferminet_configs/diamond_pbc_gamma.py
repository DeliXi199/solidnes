"""Gamma-point all-electron diamond primitive-cell config for FermiNet.

This config intentionally stays close to upstream FermiNet's PBC examples. The
SolidNES adapter applies model/training overrides from YAML experiment files.
"""

from __future__ import annotations

import numpy as np

from ferminet import base_config
from ferminet.pbc import envelopes
from ferminet.utils import system


BOHR_PER_ANGSTROM = 1.889726124565062
DIAMOND_LATTICE_CONSTANT_ANGSTROM = 3.57


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
    # FermiNet PBC routines document lattice vectors as columns.
    cartesian_bohr = fractional @ lattice.T
    return [
        system.Atom("C", tuple(cartesian_bohr[0])),
        system.Atom("C", tuple(cartesian_bohr[1])),
    ]


def get_config():
    """Build a FermiNet config for diamond PBC ground-state baselines."""

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

    # Upstream FermiNet does not currently implement HF pretraining for the PBC
    # path used here. Keep the first baseline simple and reproducible.
    cfg.pretrain.method = None
    cfg.pretrain.iterations = 0
    return cfg
