#!/usr/bin/env python
"""Run a tiny DeepSolid process smoke from a SolidNES experiment YAML."""

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
from solidnes.backends.deepsolid_compat import apply_jax_legacy_shims
from solidnes.backends.deepsolid_compat import patch_kfac_tag_primitives_for_modern_jax
from solidnes.backends.deepsolid_compat import neutralize_kfac_tags_for_smoke
from solidnes.backends.deepsolid_compat import patch_pretrain_kfac_tags_for_modern_jax
from solidnes.backends.deepsolid_compat import patch_checkpoint_save_for_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    args = parser.parse_args()

    experiment_path = (PROJECT_ROOT / args.experiment).resolve()
    bundle = build_deepsolid_adapter(experiment_path)
    cfg = bundle.cfg
    create_output_dirs(bundle)

    print(f"experiment: {bundle.experiment['experiment_name']}")
    print(f"save_path: {cfg.log.save_path}")
    print("starting DeepSolid process smoke")

    apply_jax_legacy_shims()
    if cfg.use_x64:
        import jax  # pylint: disable=import-outside-toplevel

        jax.config.update("jax_enable_x64", True)
    from DeepSolid import process  # pylint: disable=import-outside-toplevel

    if cfg.optim.optimizer != "kfac":
        neutralize_kfac_tags_for_smoke()
    else:
        patch_kfac_tag_primitives_for_modern_jax()
        patch_pretrain_kfac_tags_for_modern_jax()
    patch_checkpoint_save_for_smoke()
    process.process(cfg)

    print("DeepSolid process smoke completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
