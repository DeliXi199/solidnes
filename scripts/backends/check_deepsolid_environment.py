#!/usr/bin/env python
"""Check whether a Python environment can run basic DeepSolid config code."""

from __future__ import annotations

import importlib
import contextlib
import io
import os
from pathlib import Path
import sys


REQUIRED_MODULES = [
    "jax",
    "jaxlib",
    "pyscf",
    "ml_collections",
    "optax",
    "chex",
    "scipy",
    "numpy",
]


@contextlib.contextmanager
def _suppress_native_output():
    """Silence libraries that write directly to stdout/stderr file descriptors."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"MISSING ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.splitlines()[0]}")
    print()

    ok = True
    for name in REQUIRED_MODULES:
        version = _module_version(name)
        if version.startswith("MISSING"):
            ok = False
        print(f"{name}: {version}")

    print()
    try:
        from DeepSolid import base_config  # pylint: disable=import-outside-toplevel

        cfg = base_config.default()
        print(f"DeepSolid.base_config: OK batch_size={cfg.batch_size}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        ok = False
        print(f"DeepSolid.base_config: FAILED ({type(exc).__name__}: {exc})")

    try:
        from DeepSolid.config import diamond  # pylint: disable=import-outside-toplevel

        if os.environ.get("SOLIDNES_VERBOSE_PYSCF") == "1":
            cfg = diamond.get_config("C,C,3.57,1,ccpvdz")
        else:
            with _suppress_native_output():
                cfg = diamond.get_config("C,C,3.57,1,ccpvdz")
        cell = cfg.system.pyscf_cell
        print(
            "DeepSolid diamond config: "
            f"OK nelectron={cell.nelectron} nelec={cell.nelec}"
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        ok = False
        print(f"DeepSolid diamond config: FAILED ({type(exc).__name__}: {exc})")

    print()
    print(f"cwd: {Path.cwd()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
