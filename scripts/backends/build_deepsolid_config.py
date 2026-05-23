#!/usr/bin/env python
"""Build a DeepSolid config from a SolidNES experiment YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solidnes.backends.deepsolid_adapter import build_deepsolid_adapter
from solidnes.backends.deepsolid_adapter import create_output_dirs
from solidnes.backends.deepsolid_adapter import format_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_deepsolid_ground_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create configured DeepSolid output directories.",
    )
    args = parser.parse_args()

    experiment_path = (PROJECT_ROOT / args.experiment).resolve()
    bundle = build_deepsolid_adapter(experiment_path)

    if args.create_output_dirs:
        create_output_dirs(bundle)

    print(format_summary(bundle.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
