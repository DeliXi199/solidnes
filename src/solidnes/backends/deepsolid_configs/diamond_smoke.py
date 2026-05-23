"""Tiny carbon-diamond DeepSolid config for runtime smoke tests."""

from __future__ import annotations

import numpy as np
from pyscf.pbc import gto

from DeepSolid import base_config
from DeepSolid import supercell
from DeepSolid.utils import units


def get_config(input_str):
    """Build a deliberately small diamond config.

    Expected input:
        C,C,3.57,1,sto-3g,1e-8
    """
    x_symbol, y_symbol, lattice_angstrom, supercell_size, basis, precision = (
        input_str.split(",")
    )
    cfg = base_config.default()
    lattice_bohr = units.angstrom2bohr(float(lattice_angstrom))

    cell = gto.Cell()
    cell.atom = [
        [x_symbol, [0.0, 0.0, 0.0]],
        [
            y_symbol,
            [
                0.25 * lattice_bohr,
                0.25 * lattice_bohr,
                0.25 * lattice_bohr,
            ],
        ],
    ]
    cell.basis = basis
    cell.a = (np.ones((3, 3)) - np.eye(3)) * lattice_bohr / 2
    cell.unit = "B"
    cell.verbose = 0
    cell.precision = float(precision)
    cell.exp_to_discard = 0.1
    cell.build()

    supercell_matrix = np.eye(3) * int(supercell_size)
    cfg.system.pyscf_cell = supercell.get_supercell(cell, supercell_matrix)
    return cfg
