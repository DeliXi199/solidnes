#!/usr/bin/env python
"""GPU smoke for FermiNet PBC penalty training with real local energy."""

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
    "configs/experiment/"
    "diamond_c_ferminet_pbc_gamma_real_local_energy_training_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default=DEFAULT_EXPERIMENT,
        help="SolidNES FermiNet experiment YAML to build.",
    )
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument(
        "--walkers",
        type=int,
        default=None,
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--penalty-alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
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
    experiment_path = _resolve_project_path(args.experiment)
    _apply_runtime_defaults(experiment_path, platform=args.platform)

    from solidnes.backends.deepsolid_adapter import load_yaml
    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        assert_pbc_external_state_config,
        build_external_state_adapter,
    )
    from solidnes.excited_states.ferminet_pbc_training import (
        run_external_state_penalty_sgd,
    )

    experiment = load_yaml(experiment_path)
    _resolve_runtime_args(args, experiment)
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 1:
        raise ValueError("--walkers must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")

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
    result = run_external_state_penalty_sgd(
        adapter,
        params,
        samples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        penalty_alpha=args.penalty_alpha,
        block_until_ready=True,
    )
    elapsed_seconds = time.perf_counter() - start

    _validate_terms(adapter, "initial", result.initial_terms, args)
    _validate_terms(adapter, "final", result.final_terms, args)
    for diagnostics in result.history:
        step = diagnostics.step
        _assert_finite(
            adapter,
            f"penalty_objective_step_{step}",
            diagnostics.penalty_objective,
        )
        _assert_finite(
            adapter,
            f"gradient_objective_step_{step}",
            diagnostics.gradient_objective,
        )
        _assert_positive_finite(
            adapter,
            f"grad_l2_norm_step_{step}",
            diagnostics.grad_l2_norm,
        )
        _assert_positive_finite(
            adapter,
            f"param_delta_l2_norm_step_{step}",
            diagnostics.param_delta_l2_norm,
        )
        _assert_all_finite(
            adapter,
            f"state_energy_step_{step}",
            diagnostics.state_energy,
        )
        _assert_all_finite(
            adapter,
            f"offdiag_squared_overlap_step_{step}",
            diagnostics.offdiag_squared_overlap,
        )
        _assert_bool_true(adapter, f"gradient_finite_step_{step}", diagnostics.gradient_finite)
        _assert_bool_true(adapter, f"update_finite_step_{step}", diagnostics.update_finite)
        _assert_bool_true(
            adapter,
            f"candidate_terms_finite_step_{step}",
            diagnostics.candidate_terms_finite,
        )
        _assert_bool_true(
            adapter,
            f"gradient_objective_finite_step_{step}",
            diagnostics.gradient_objective_finite,
        )
        _assert_bool_true(adapter, f"update_accepted_step_{step}", diagnostics.update_accepted)

    output_dir = _resolve_output_dir(args, experiment)
    initial_objective = result.initial_terms["penalty_objective"]
    final_objective = result.final_terms["penalty_objective"]
    summary = {
        "status": "ok",
        "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cfg_system_states": cfg_states,
        "external_state_params": args.states,
        "walkers_per_state": args.walkers,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "gradient_mode": "paper_tangent",
        "local_energy_source": "real_pbc",
        "penalty_alpha": args.penalty_alpha,
        "elapsed_seconds": elapsed_seconds,
        "initial_penalty_objective": _json_value(initial_objective),
        "final_penalty_objective": _json_value(final_objective),
        "objective_delta": _json_value(final_objective - initial_objective),
        "initial_state_energy": _json_value(result.initial_terms["state_energy"]),
        "final_state_energy": _json_value(result.final_terms["state_energy"]),
        "final_overlap_matrix": _json_value(result.final_terms["overlap_matrix"]),
        "history": [_diagnostics_summary(item) for item in result.history],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "real_local_energy_training_smoke_summary.json"
        md_path = output_dir / "real_local_energy_training_smoke_summary.md"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown_summary(summary), encoding="utf-8")
        summary["summary_json"] = str(json_path.relative_to(PROJECT_ROOT))
        summary["summary_markdown"] = str(md_path.relative_to(PROJECT_ROOT))

    print("ferminet_pbc_real_local_energy_training_smoke: ok")
    print(f"experiment: {summary['experiment']}")
    print(f"jax_platform: {summary['jax_platform']}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print(f"walkers_per_state: {args.walkers}")
    print(f"steps: {args.steps}")
    print(f"learning_rate: {args.learning_rate:.12g}")
    print("gradient_mode: paper_tangent")
    print("local_energy_source: real_pbc")
    print(f"elapsed_seconds: {elapsed_seconds:.6f}")
    print(f"initial_penalty_objective: {float(initial_objective):.12g}")
    print(f"final_penalty_objective: {float(final_objective):.12g}")
    print(f"objective_delta: {float(final_objective - initial_objective):.12g}")
    for item in result.history:
        print(
            f"step_{item.step}: objective={float(item.penalty_objective):.12g} "
            f"gradient_objective={float(item.gradient_objective):.12g} "
            f"grad_l2_norm={float(item.grad_l2_norm):.12g} "
            f"param_delta_l2_norm={float(item.param_delta_l2_norm):.12g} "
            f"update_accepted={bool(item.update_accepted)}"
        )
    if output_dir is not None:
        print(f"summary_json: {summary['summary_json']}")
        print(f"summary_markdown: {summary['summary_markdown']}")
    return 0


def _validate_terms(
    adapter: Any,
    label: str,
    terms: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    expected_shapes = {
        "local_energy": (args.states, args.walkers),
        "state_energy": (args.states,),
        "state_energy_std": (args.states,),
        "overlap_matrix": (args.states, args.states),
        "overlap_gradient_scale": (args.states, args.states),
        "psi_ratio": (args.states, args.states, args.walkers),
        "psi_ratio_for_overlap": (args.states, args.states, args.walkers),
        "penalty_objective": (),
    }
    for name, shape in expected_shapes.items():
        observed = tuple(adapter.modules.jnp.asarray(terms[name]).shape)
        if observed != shape:
            raise ValueError(
                f"{label}_{name} shape mismatch: expected {shape}, got {observed}"
            )
        _assert_all_finite(adapter, f"{label}_{name}", terms[name])


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


def _resolve_runtime_args(args: argparse.Namespace, experiment: dict[str, Any]) -> None:
    diagnostics = experiment.get("diagnostics", {})
    args.states = _resolve_int_option(
        args.states,
        env_name="SOLIDNES_NES_STATES",
        config_value=diagnostics.get("expected_external_states"),
        default=2,
    )
    args.walkers = _resolve_int_option(
        args.walkers,
        env_name="SOLIDNES_NES_WALKERS",
        config_value=diagnostics.get("walkers_per_state"),
        default=1,
    )
    args.steps = _resolve_int_option(
        args.steps,
        env_name="SOLIDNES_NES_STEPS",
        config_value=diagnostics.get("steps"),
        default=2,
    )
    args.seed = _resolve_int_option(
        args.seed,
        env_name="SOLIDNES_NES_SEED",
        config_value=diagnostics.get("seed"),
        default=47,
    )
    args.penalty_alpha = _resolve_float_option(
        args.penalty_alpha,
        env_name="SOLIDNES_NES_PENALTY_ALPHA",
        config_value=diagnostics.get("penalty_alpha"),
        default=1.0,
    )
    args.learning_rate = _resolve_float_option(
        args.learning_rate,
        env_name="SOLIDNES_NES_LEARNING_RATE",
        config_value=diagnostics.get("learning_rate"),
        default=1.0e-8,
    )


def _resolve_int_option(
    value: int | None,
    *,
    env_name: str,
    config_value: Any,
    default: int,
) -> int:
    if value is not None:
        return int(value)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value)
    if config_value is not None:
        return int(config_value)
    return default


def _resolve_float_option(
    value: float | None,
    *,
    env_name: str,
    config_value: Any,
    default: float,
) -> float:
    if value is not None:
        return float(value)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return float(env_value)
    if config_value is not None:
        return float(config_value)
    return default


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


def _assert_finite(adapter: Any, name: str, value: Any) -> None:
    if not bool(adapter.modules.jnp.isfinite(value)):
        raise ValueError(f"{name} is not finite: {value}")


def _assert_positive_finite(adapter: Any, name: str, value: Any) -> None:
    _assert_finite(adapter, name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _assert_bool_true(adapter: Any, name: str, value: Any) -> None:
    del adapter
    if not bool(value):
        raise ValueError(f"{name} must be true, got {value}")


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


def _diagnostics_summary(diagnostics: Any) -> dict[str, Any]:
    return {
        "step": int(diagnostics.step),
        "penalty_objective": _json_value(diagnostics.penalty_objective),
        "gradient_objective": _json_value(diagnostics.gradient_objective),
        "weighted_state_energy": _json_value(diagnostics.weighted_state_energy),
        "state_energy": _json_value(diagnostics.state_energy),
        "state_energy_std": _json_value(diagnostics.state_energy_std),
        "offdiag_squared_overlap": _json_value(
            diagnostics.offdiag_squared_overlap
        ),
        "scaled_offdiag_squared_overlap": _json_value(
            diagnostics.scaled_offdiag_squared_overlap
        ),
        "overlap_penalty": _json_value(diagnostics.overlap_penalty),
        "overlap_gradient_scale": _json_value(diagnostics.overlap_gradient_scale),
        "max_abs_offdiag_overlap": _json_value(
            diagnostics.max_abs_offdiag_overlap
        ),
        "collapse_flag": _json_value(diagnostics.collapse_flag),
        "grad_l2_norm": _json_value(diagnostics.grad_l2_norm),
        "param_delta_l2_norm": _json_value(diagnostics.param_delta_l2_norm),
        "gradient_objective_finite": _json_value(
            diagnostics.gradient_objective_finite
        ),
        "gradient_finite": _json_value(diagnostics.gradient_finite),
        "update_finite": _json_value(diagnostics.update_finite),
        "candidate_terms_finite": _json_value(diagnostics.candidate_terms_finite),
        "update_accepted": _json_value(diagnostics.update_accepted),
    }


def _format_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# FermiNet PBC Real Local-Energy Training Smoke",
        "",
        "```text",
        f"status: {summary['status']}",
        f"experiment: {summary['experiment']}",
        f"jax_platform: {summary['jax_platform']}",
        f"cfg_system_states: {summary['cfg_system_states']}",
        f"external_state_params: {summary['external_state_params']}",
        f"walkers_per_state: {summary['walkers_per_state']}",
        f"steps: {summary['steps']}",
        f"learning_rate: {summary['learning_rate']:.12g}",
        f"gradient_mode: {summary['gradient_mode']}",
        f"local_energy_source: {summary['local_energy_source']}",
        f"elapsed_seconds: {summary['elapsed_seconds']:.6f}",
        f"initial_penalty_objective: {summary['initial_penalty_objective']}",
        f"final_penalty_objective: {summary['final_penalty_objective']}",
        f"objective_delta: {summary['objective_delta']}",
        f"initial_state_energy: {summary['initial_state_energy']}",
        f"final_state_energy: {summary['final_state_energy']}",
        "```",
        "",
        "Per-step diagnostics:",
        "",
    ]
    for item in summary["history"]:
        lines.extend(
            [
                "```text",
                f"step: {item['step']}",
                f"penalty_objective: {item['penalty_objective']}",
                f"gradient_objective: {item['gradient_objective']}",
                f"state_energy: {item['state_energy']}",
                f"overlap_penalty: {item['overlap_penalty']}",
                f"overlap_gradient_scale: {item['overlap_gradient_scale']}",
                f"grad_l2_norm: {item['grad_l2_norm']}",
                f"param_delta_l2_norm: {item['param_delta_l2_norm']}",
                f"gradient_objective_finite: {item['gradient_objective_finite']}",
                f"gradient_finite: {item['gradient_finite']}",
                f"update_finite: {item['update_finite']}",
                f"candidate_terms_finite: {item['candidate_terms_finite']}",
                f"update_accepted: {item['update_accepted']}",
                f"collapse_flag: {item['collapse_flag']}",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "This smoke runs the reusable fixed-sample external-state penalty",
            "training loop with the real FermiNet PBC local-energy/Laplacian",
            "path. It checks finite diagnostics and nonzero parameter updates,",
            "not variational convergence.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
