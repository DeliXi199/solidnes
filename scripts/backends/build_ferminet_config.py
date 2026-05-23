#!/usr/bin/env python
"""Build a FermiNet config from a SolidNES experiment YAML."""

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

from solidnes.backends.ferminet_adapter import build_ferminet_adapter
from solidnes.backends.ferminet_adapter import create_output_dirs
from solidnes.backends.ferminet_adapter import format_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create configured FermiNet output directories.",
    )
    args = parser.parse_args()

    if os.environ.get("SOLIDNES_REQUIRE_JAX_GPU", "0") not in {"1", "true"}:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    experiment_path = (PROJECT_ROOT / args.experiment).resolve()
    bundle = build_ferminet_adapter(experiment_path)
    if args.create_output_dirs:
        create_output_dirs(bundle)
    print(format_summary(bundle.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
