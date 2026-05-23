"""Compatibility wrapper for upstream FermiNet PBC Hamiltonian."""

from __future__ import annotations

from typing import Any

from ferminet.pbc import hamiltonian as pbc_hamiltonian


def local_energy(*args: Any, ndim: int | None = None, **kwargs: Any):
    """Call upstream PBC local_energy while accepting train.py's ndim kwarg."""

    del ndim
    return pbc_hamiltonian.local_energy(*args, **kwargs)

