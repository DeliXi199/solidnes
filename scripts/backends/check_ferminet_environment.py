#!/usr/bin/env python
"""Check whether a Python environment can run basic FermiNet PBC config code."""

from __future__ import annotations

import argparse
import importlib
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
from solidnes.backends.ferminet_adapter import format_summary


REQUIRED_MODULES = [
    "jax",
    "jaxlib",
    "folx",
    "kfac_jax",
    "pyscf",
    "ml_collections",
    "optax",
    "chex",
    "scipy",
    "numpy",
    "ferminet",
]


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"MISSING ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=os.environ.get("SOLIDNES_REQUIRE_JAX_GPU", "0") in {"1", "true"},
        help="Fail unless JAX reports at least one GPU.",
    )
    parser.add_argument(
        "--expected-jax-version",
        default=os.environ.get("SOLIDNES_EXPECT_JAX_VERSION", "0.10.1"),
        help="Expected exact JAX version for the latest-JAX baseline.",
    )
    args = parser.parse_args()
    if not args.require_gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print(f"python: {sys.executable}")
    print(f"version: {sys.version.splitlines()[0]}")
    print(f"cwd: {Path.cwd()}")
    print()

    ok = True
    versions: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        version = _module_version(name)
        versions[name] = version
        if version.startswith("MISSING"):
            ok = False
        print(f"{name}: {version}")

    if versions.get("jax") != args.expected_jax_version:
        ok = False
        print(
            "jax_version_check: FAILED "
            f"expected={args.expected_jax_version} observed={versions.get('jax')}"
        )
    else:
        print(f"jax_version_check: OK expected={args.expected_jax_version}")

    print()
    try:
        import jax  # pylint: disable=import-outside-toplevel

        print(f"jax_default_backend: {jax.default_backend()}")
        print(f"jax_devices: {jax.devices()}")
        print(f"jax_enable_x64: {jax.config.jax_enable_x64}")
        if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
            ok = False
            print("jax_gpu_check: FAILED no GPU device reported")
        else:
            print("jax_gpu_check: OK")
    except Exception as exc:  # pragma: no cover - diagnostic script
        ok = False
        print(f"jax_runtime_check: FAILED ({type(exc).__name__}: {exc})")

    print()
    try:
        bundle = build_ferminet_adapter(PROJECT_ROOT / args.experiment)
        print("FermiNet diamond PBC config: OK")
        print(format_summary(bundle.summary))
    except Exception as exc:  # pragma: no cover - diagnostic script
        ok = False
        print(f"FermiNet diamond PBC config: FAILED ({type(exc).__name__}: {exc})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
