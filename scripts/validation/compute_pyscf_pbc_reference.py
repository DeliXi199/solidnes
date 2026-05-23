#!/usr/bin/env python
"""Compute a PySCF PBC Hartree-Fock reference for a SolidNES experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solidnes.backends.deepsolid_adapter import build_deepsolid_adapter
from solidnes.backends.deepsolid_adapter import suppress_native_output


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _default_output_path(bundle) -> Path:
    output = bundle.experiment.get("output", {})
    validation_dir = output.get("validation_dir")
    if validation_dir:
        root = PROJECT_ROOT / validation_dir
    else:
        root = Path(bundle.cfg.log.save_path).parent / "validation"
    return root / "pyscf_pbc_hf_reference.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_deepsolid_validation_short.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to the experiment validation directory.",
    )
    parser.add_argument(
        "--verbose-pyscf",
        action="store_true",
        help="Show PySCF native output.",
    )
    args = parser.parse_args()

    bundle = build_deepsolid_adapter(PROJECT_ROOT / args.experiment)
    cfg = bundle.cfg
    cell = cfg.system.pyscf_cell

    from DeepSolid import hf  # pylint: disable=import-outside-toplevel

    twist = np.asarray(cfg.network.twist)
    if args.verbose_pyscf:
        scf_ref = hf.SCF(cell=cell, twist=twist)
        scf_ref.init_scf()
    else:
        with suppress_native_output():
            scf_ref = hf.SCF(cell=cell, twist=twist)
            scf_ref.init_scf()

    kmf = scf_ref.kmf
    total_energy = float(np.real(kmf.e_tot))
    scale = int(getattr(cell, "scale", 1))
    basis_scope = f"same configured {bundle.summary.basis} Hamiltonian"
    result = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": bundle.experiment["experiment_name"],
        "backend": bundle.experiment["backend"]["name"],
        "reference": "PySCF PBC KHF via DeepSolid.hf.SCF",
        "basis": bundle.summary.basis,
        "pseudo": bundle.summary.pseudo,
        "nelectron": int(cell.nelectron),
        "nelec": list(cell.nelec),
        "dimension": int(cell.dimension),
        "scale": scale,
        "twist": twist.tolist(),
        "kpts": _json_ready(kmf.kpts),
        "converged": bool(kmf.converged),
        "e_tot_hartree": total_energy,
        "e_tot_per_simulation_cell_hartree": total_energy / scale,
        "e_tot_per_electron_hartree": total_energy / float(cell.nelectron),
        "mo_occ_shape": _json_ready(np.asarray(kmf.mo_occ).shape),
        "notes": {
            "accuracy_scope": basis_scope,
            "comparison_rule": "a trained variational energy should be finite and trend below/near this HF reference for the same setup",
        },
    }

    output_path = Path(args.output).resolve() if args.output else _default_output_path(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_ready)
        handle.write("\n")

    print(f"experiment: {result['experiment_name']}")
    print(f"reference: {result['reference']}")
    print(f"converged: {result['converged']}")
    print(f"e_tot_hartree: {result['e_tot_hartree']:.12g}")
    print(f"e_tot_per_electron_hartree: {result['e_tot_per_electron_hartree']:.12g}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
