#!/usr/bin/env python
"""Run FermiNet from a SolidNES experiment YAML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


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
from solidnes.backends.ferminet_checkpoint_policy import (
    assert_final_checkpoint_written,
    enforce_ferminet_final_checkpoint,
)
from solidnes.backends.ferminet_jax_compat import apply_modern_jax_shims
from solidnes.backends.ferminet_psiformer_attention import (
    install_psiformer_attention_implementation,
    resolved_psiformer_attention_kernel_gpu,
)


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

    from absl import logging as absl_logging  # pylint: disable=import-outside-toplevel

    absl_logging.set_verbosity(absl_logging.INFO)
    resolved_attention = install_psiformer_attention_implementation(bundle.cfg)
    if resolved_attention is not None:
        print(f"psiformer_attention_resolved: {resolved_attention}")
        print(
            "psiformer_attention_kernel_gpu_resolved: "
            f"{resolved_psiformer_attention_kernel_gpu()}"
        )
    from ferminet import train  # pylint: disable=import-outside-toplevel

    start = time.perf_counter()
    with enforce_ferminet_final_checkpoint(train, bundle.cfg) as final_checkpoint:
        if final_checkpoint.required:
            print(f"final_checkpoint_required: {final_checkpoint.checkpoint_path}")
        train.train(bundle.cfg)
    assert_final_checkpoint_written(final_checkpoint)
    elapsed_seconds = time.perf_counter() - start
    _write_runtime_metadata(
        bundle,
        elapsed_seconds=elapsed_seconds,
        resolved_attention=resolved_attention,
        jax_version=jax.__version__,
        jax_backend=jax.default_backend(),
        jax_devices=[str(device) for device in jax.devices()],
        final_checkpoint=final_checkpoint.to_json(),
    )
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


def _write_runtime_metadata(
    bundle,
    *,
    elapsed_seconds: float,
    resolved_attention: str | None,
    jax_version: str,
    jax_backend: str,
    jax_devices: list[str],
    final_checkpoint: dict[str, object],
) -> None:
    output = bundle.experiment.get("output", {})
    validation_dir = output.get("validation_dir")
    if not validation_dir:
        return
    validation_path = PROJECT_ROOT / validation_dir
    validation_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": bundle.summary.experiment_name,
        "network_type": bundle.summary.network_type,
        "objective": bundle.summary.objective,
        "states": bundle.summary.states,
        "iterations": bundle.summary.iterations,
        "batch_size": bundle.summary.batch_size,
        "optimizer": bundle.summary.optimizer,
        "laplacian": bundle.summary.laplacian,
        "psiformer_attention_configured": (
            bundle.summary.psiformer_attention_implementation
        ),
        "psiformer_attention_resolved": resolved_attention,
        "psiformer_attention_kernel_gpu": bundle.summary.psiformer_attention_kernel_gpu,
        "psiformer_attention_kernel_gpu_resolved": (
            resolved_psiformer_attention_kernel_gpu()
            if resolved_attention is not None
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_iteration": (
            elapsed_seconds / bundle.summary.iterations
            if bundle.summary.iterations > 0
            else None
        ),
        "jax_version": jax_version,
        "jax_backend": jax_backend,
        "jax_devices": jax_devices,
        "final_checkpoint": final_checkpoint,
    }
    path = validation_path / "ferminet_train_runtime.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"runtime_metadata: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
