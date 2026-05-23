#!/usr/bin/env python
"""Probe DeepSolid runtime objects exposed by the SolidNES adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solidnes.backends.deepsolid_adapter import build_deepsolid_adapter
from solidnes.backends.deepsolid_adapter import create_output_dirs
from solidnes.backends.deepsolid_adapter import initialize_deepsolid_ground_state


def _as_array(value: Any) -> np.ndarray:
    import jax  # pylint: disable=import-outside-toplevel

    return np.asarray(jax.device_get(value))


def _first_scalar(value: Any) -> Any:
    return _as_array(value).reshape(-1)[0].item()


def _mean_scalar(value: Any) -> Any:
    return _as_array(value).mean().item()


def _format_scalar(value: Any) -> str:
    scalar = complex(value) if isinstance(value, np.complexfloating) else value
    if isinstance(scalar, complex):
        if abs(scalar.imag) < 1.0e-12:
            return f"{scalar.real:.12g}"
        return f"{scalar.real:.12g}{scalar.imag:+.12g}j"
    return f"{float(scalar):.12g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--skip-energy",
        action="store_true",
        help="Only initialize objects and skip the local-energy call.",
    )
    parser.add_argument(
        "--skip-mcmc",
        action="store_true",
        help="Only initialize/evaluate energy and skip the MCMC step.",
    )
    args = parser.parse_args()

    experiment_path = (PROJECT_ROOT / args.experiment).resolve()
    bundle = build_deepsolid_adapter(experiment_path)
    create_output_dirs(bundle)

    print(f"experiment: {bundle.experiment['experiment_name']}")
    print(f"basis: {bundle.summary.basis}")
    print(f"pseudo: {bundle.summary.pseudo}")
    print(f"nelectron: {bundle.summary.nelectron}")
    print(f"nelec: {bundle.summary.nelec}")

    objects = initialize_deepsolid_ground_state(bundle)
    print("runtime_objects: initialized")
    print(f"num_devices: {objects.num_devices}")
    print(f"local_batch_size: {objects.local_batch_size}")
    print(f"data_shape: {tuple(objects.data.shape)}")

    import jax  # pylint: disable=import-outside-toplevel

    param_leaves = jax.tree_util.tree_leaves(objects.params)
    print(f"param_leaf_count: {len(param_leaves)}")
    if param_leaves:
        print(f"first_param_shape: {tuple(param_leaves[0].shape)}")

    if not args.skip_energy:
        loss, aux = objects.evaluate_energy()
        scale = objects.simulation_cell.scale
        print(f"energy: {_format_scalar(_first_scalar(loss) / scale)}")
        print(f"variance: {_format_scalar(_first_scalar(aux.variance) / (scale**2))}")
        print(f"imaginary: {_format_scalar(_first_scalar(aux.imaginary) / scale)}")
        print(f"kinetic_mean: {_format_scalar(_mean_scalar(aux.kinetic) / scale)}")
        print(f"ewald_mean: {_format_scalar(_mean_scalar(aux.ewald) / scale)}")

    if not args.skip_mcmc:
        data, pmove = objects.run_mcmc_step()
        print(f"mcmc_data_shape: {tuple(data.shape)}")
        print(f"pmove: {_format_scalar(_first_scalar(pmove))}")

    print("DeepSolid adapter object probe completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
