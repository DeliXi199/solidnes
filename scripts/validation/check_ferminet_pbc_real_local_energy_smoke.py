#!/usr/bin/env python
"""GPU smoke for FermiNet PBC penalty terms with real local energy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    "configs/experiment/diamond_c_ferminet_pbc_gamma_real_local_energy_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default=DEFAULT_EXPERIMENT,
        help="SolidNES FermiNet experiment YAML to build.",
    )
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--penalty-alpha", type=float, default=1.0)
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional JAX platform override. Leave unset for scheduled GPU.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for JSON and Markdown validation summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 1:
        raise ValueError("--walkers must be at least 1")

    experiment_path = _resolve_project_path(args.experiment)
    _apply_runtime_defaults(experiment_path, platform=args.platform)

    from solidnes.backends.deepsolid_adapter import load_yaml
    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        assert_pbc_external_state_config,
        build_external_state_adapter,
        evaluate_ferminet_pbc_penalty_terms,
    )

    experiment = load_yaml(experiment_path)
    bundle = build_ferminet_adapter(experiment_path)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)
    jax = adapter.modules.jax

    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key = jax.random.split(key, 3)
    params = adapter.init_state_params(params_key, args.states)
    samples = adapter.tiny_state_samples(
        samples_key,
        nstates=args.states,
        walkers=args.walkers,
    )

    start = time.perf_counter()
    terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        samples,
        penalty_alpha=args.penalty_alpha,
    )
    terms = _block_until_ready(jax, terms)
    elapsed_seconds = time.perf_counter() - start

    expected_shapes = {
        "local_energy": (args.states, args.walkers),
        "state_energy": (args.states,),
        "overlap_matrix": (args.states, args.states),
        "psi_ratio": (args.states, args.states, args.walkers),
        "penalty_objective": (),
    }
    for name, shape in expected_shapes.items():
        observed = tuple(adapter.modules.jnp.asarray(terms[name]).shape)
        if observed != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {observed}")
        _assert_all_finite(adapter, name, terms[name])

    output_dir = _resolve_output_dir(args, experiment)
    summary = {
        "status": "ok",
        "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cfg_system_states": cfg_states,
        "external_state_params": args.states,
        "walkers_per_state": args.walkers,
        "local_energy_source": "real_pbc",
        "penalty_alpha": args.penalty_alpha,
        "elapsed_seconds": elapsed_seconds,
        "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
        "local_energy": _json_value(terms["local_energy"]),
        "state_energy": _json_value(terms["state_energy"]),
        "overlap_matrix": _json_value(terms["overlap_matrix"]),
        "offdiag_squared_overlap": _json_value(
            terms["offdiag_squared_overlap"]
        ),
        "penalty_objective": _json_value(terms["penalty_objective"]),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "real_local_energy_smoke_summary.json"
        md_path = output_dir / "real_local_energy_smoke_summary.md"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown_summary(summary), encoding="utf-8")
        summary["summary_json"] = str(json_path.relative_to(PROJECT_ROOT))
        summary["summary_markdown"] = str(md_path.relative_to(PROJECT_ROOT))

    print("ferminet_pbc_real_local_energy_smoke: ok")
    print(f"experiment: {summary['experiment']}")
    print(f"jax_platform: {summary['jax_platform']}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print(f"walkers_per_state: {args.walkers}")
    print("local_energy_source: real_pbc")
    print(f"elapsed_seconds: {elapsed_seconds:.6f}")
    print(f"local_energy_shape: {expected_shapes['local_energy']}")
    print(f"state_energy_shape: {expected_shapes['state_energy']}")
    print(f"overlap_matrix_shape: {expected_shapes['overlap_matrix']}")
    print(f"psi_ratio_shape: {expected_shapes['psi_ratio']}")
    print(f"penalty_objective_shape: {expected_shapes['penalty_objective']}")
    print(f"state_energy: {summary['state_energy']}")
    print(f"penalty_objective: {summary['penalty_objective']}")
    if output_dir is not None:
        print(f"summary_json: {summary['summary_json']}")
        print(f"summary_markdown: {summary['summary_markdown']}")
    return 0


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _apply_runtime_defaults(experiment_path: Path, *, platform: str | None) -> None:
    from solidnes.backends.deepsolid_adapter import load_yaml

    runtime = load_yaml(experiment_path).get("runtime", {})
    if bool(runtime.get("x64_enabled", False)):
        os.environ["JAX_ENABLE_X64"] = "1"
    else:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    if platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)


def _block_until_ready(jax: Any, tree: Any) -> Any:
    def block(value: Any) -> Any:
        if hasattr(value, "block_until_ready"):
            return value.block_until_ready()
        return value

    return jax.tree_util.tree_map(block, tree)


def _assert_all_finite(adapter: Any, name: str, value: Any) -> None:
    jnp = adapter.modules.jnp
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        finite = jnp.all(jnp.isfinite(jnp.real(array))) & jnp.all(
            jnp.isfinite(jnp.imag(array))
        )
    else:
        finite = jnp.all(jnp.isfinite(array))
    if not bool(finite):
        raise ValueError(f"{name} contains non-finite values: {value}")


def _resolve_output_dir(args: argparse.Namespace, experiment: dict[str, Any]) -> Path | None:
    if args.output_dir:
        return _resolve_project_path(args.output_dir)
    task_root = os.environ.get("SOLIDNES_TASK_ROOT")
    if task_root:
        return _resolve_project_path(task_root) / "results" / "validation"
    validation_dir = experiment.get("output", {}).get("validation_dir")
    if validation_dir:
        return _resolve_project_path(validation_dir)
    return None


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "real": np.real(array).tolist(),
            "imag": np.imag(array).tolist(),
        }
    return array.tolist()


def _format_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# FermiNet PBC Real Local-Energy Smoke",
        "",
        "```text",
        f"status: {summary['status']}",
        f"experiment: {summary['experiment']}",
        f"jax_platform: {summary['jax_platform']}",
        f"cfg_system_states: {summary['cfg_system_states']}",
        f"external_state_params: {summary['external_state_params']}",
        f"walkers_per_state: {summary['walkers_per_state']}",
        f"local_energy_source: {summary['local_energy_source']}",
        f"elapsed_seconds: {summary['elapsed_seconds']:.6f}",
        f"state_energy: {summary['state_energy']}",
        f"penalty_objective: {summary['penalty_objective']}",
        "```",
        "",
        "This smoke evaluates the real FermiNet PBC local-energy/Laplacian path",
        "for two externally managed state parameter trees on a one-walker",
        "carbon-diamond Gamma sample per state.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
