#!/usr/bin/env python
"""Run FermiNet from a SolidNES experiment YAML."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (SRC_ROOT, FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from solidnes.backends.deepsolid_adapter import load_yaml
from solidnes.backends.ferminet_adapter import build_ferminet_adapter
from solidnes.backends.ferminet_adapter import create_output_dirs
from solidnes.backends.ferminet_adapter import format_summary
from solidnes.backends.ferminet_jax_compat import apply_modern_jax_shims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and summarize the config without starting FermiNet training.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when JAX reports only CPU devices.",
    )
    args = parser.parse_args()

    experiment_path = (PROJECT_ROOT / args.experiment).resolve()
    _apply_runtime_defaults(
        experiment_path,
        force_cpu=args.allow_cpu or args.build_only,
    )
    bundle = build_ferminet_adapter(experiment_path)
    create_output_dirs(bundle)
    print(format_summary(bundle.summary))

    if args.build_only:
        return 0

    import jax  # pylint: disable=import-outside-toplevel

    apply_modern_jax_shims()
    print(f"jax={jax.__version__}")
    print(f"jax_default_backend={jax.default_backend()}")
    print(f"jax_devices={jax.devices()}")
    if not args.allow_cpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise SystemExit("JAX did not report a GPU device; pass --allow-cpu only for local debugging")

    from ferminet import train  # pylint: disable=import-outside-toplevel

    train.train(bundle.cfg)
    return 0


def _apply_runtime_defaults(experiment_path: Path, *, force_cpu: bool = False) -> None:
    """Set runtime defaults before JAX is imported."""

    runtime = load_yaml(experiment_path).get("runtime", {})
    if bool(runtime.get("x64_enabled", False)):
        os.environ["JAX_ENABLE_X64"] = "1"
    else:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    if force_cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


if __name__ == "__main__":
    raise SystemExit(main())
