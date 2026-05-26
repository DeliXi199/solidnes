#!/usr/bin/env python
"""Build-only smoke for the FermiNet PBC excited-state driver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    "configs/experiment/"
    "diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml"
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
    parser.add_argument("--walkers", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--sampler-steps", type=int, default=None)
    parser.add_argument("--proposal-width", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--penalty-alpha", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", dest="optimizer_name", default=None)
    parser.add_argument("--adam-b1", type=float, default=None)
    parser.add_argument("--adam-b2", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lamb-eps", type=float, default=None)
    parser.add_argument("--kfac-damping", type=float, default=None)
    parser.add_argument("--kfac-momentum", type=float, default=None)
    parser.add_argument("--kfac-norm-constraint", type=float, default=None)
    parser.add_argument("--kfac-l2-reg", type=float, default=None)
    parser.add_argument("--kfac-cov-ema-decay", type=float, default=None)
    parser.add_argument("--kfac-invert-every", type=int, default=None)
    parser.add_argument("--kfac-register-only-generic", action="store_true")
    parser.add_argument("--max-grad-l2-norm", type=float, default=None)
    parser.add_argument("--max-update-l2-norm", type=float, default=None)
    parser.add_argument("--overlap-ewma-decay", type=float, default=None)
    parser.add_argument("--param-share-keys", default=None)
    parser.add_argument("--candidate-check-period", type=int, default=None)
    parser.add_argument(
        "--local-energy-source",
        choices=("cheap", "real_pbc"),
        default=None,
        help="Use a cheap stand-in or the real FermiNet PBC local energy.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional JAX platform override. Leave unset for scheduled GPU.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for JSON/Markdown driver summaries.",
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
    from solidnes.excited_states.ferminet_pbc_driver import (
        run_external_state_penalty_driver,
    )

    experiment = load_yaml(experiment_path)
    _resolve_runtime_args(args, experiment)
    _validate_args(args)

    bundle = build_ferminet_adapter(experiment_path)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)
    if args.local_energy_source == "cheap":
        local_energy = _cheap_local_energy(adapter)
    else:
        local_energy = None

    jax = adapter.modules.jax
    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key, sampler_key = jax.random.split(key, 4)
    params = adapter.init_state_params(params_key, args.states)
    samples = adapter.tiny_state_samples(
        samples_key,
        nstates=args.states,
        walkers=args.walkers,
    )

    result = run_external_state_penalty_driver(
        adapter,
        params,
        samples,
        sampler_key=sampler_key,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        penalty_alpha=args.penalty_alpha,
        sampler_burn_in=args.burn_in,
        sampler_steps_per_iteration=args.sampler_steps,
        sampler_proposal_width=args.proposal_width,
        local_energy=local_energy,
        optimizer_name=args.optimizer_name,
        adam_b1=args.adam_b1,
        adam_b2=args.adam_b2,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        lamb_eps=args.lamb_eps,
        kfac_damping=args.kfac_damping,
        kfac_momentum=args.kfac_momentum,
        kfac_norm_constraint=args.kfac_norm_constraint,
        kfac_l2_reg=args.kfac_l2_reg,
        kfac_cov_ema_decay=args.kfac_cov_ema_decay,
        kfac_invert_every=args.kfac_invert_every,
        kfac_register_only_generic=args.kfac_register_only_generic or None,
        overlap_ewma_decay=args.overlap_ewma_decay,
        max_grad_l2_norm=args.max_grad_l2_norm,
        max_update_l2_norm=args.max_update_l2_norm,
        param_share_keys=args.param_share_keys,
        candidate_check_period=args.candidate_check_period,
        block_until_ready=True,
    )
    _validate_result(adapter, result, args)
    checkpoint_size = _roundtrip_checkpoint(
        adapter,
        result,
        metadata={"experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT))},
    )

    summary = {
        "status": "ok",
        "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cfg_system_states": cfg_states,
        "external_state_params": args.states,
        "walkers_per_state": args.walkers,
        "iterations": args.iterations,
        "burn_in": args.burn_in,
        "sampler_steps_per_iteration": args.sampler_steps,
        "proposal_width": args.proposal_width,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer_name,
        "kfac_damping": args.kfac_damping,
        "kfac_momentum": args.kfac_momentum,
        "kfac_norm_constraint": args.kfac_norm_constraint,
        "kfac_l2_reg": args.kfac_l2_reg,
        "kfac_cov_ema_decay": args.kfac_cov_ema_decay,
        "kfac_invert_every": args.kfac_invert_every,
        "kfac_register_only_generic": args.kfac_register_only_generic,
        "overlap_ewma_decay": args.overlap_ewma_decay,
        "param_share_keys": list(args.param_share_keys),
        "candidate_check_period": args.candidate_check_period,
        "gradient_mode": "paper_tangent",
        "local_energy_source": args.local_energy_source,
        "checkpoint_roundtrip_bytes": checkpoint_size,
        "initial_penalty_objective": _json_value(result.initial_terms["penalty_objective"]),
        "final_penalty_objective": _json_value(result.final_terms["penalty_objective"]),
        "history": [_history_item(item) for item in result.history],
    }
    output_dir = _resolve_output_dir(args, experiment)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "ferminet_pbc_driver_smoke_summary.json"
        md_path = output_dir / "ferminet_pbc_driver_smoke_summary.md"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown_summary(summary), encoding="utf-8")
        summary["summary_json"] = _display_path(json_path)
        summary["summary_markdown"] = _display_path(md_path)

    print("ferminet_pbc_driver_smoke: ok")
    print(f"experiment: {summary['experiment']}")
    print(f"jax_platform: {summary['jax_platform']}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print(f"walkers_per_state: {args.walkers}")
    print(f"iterations: {args.iterations}")
    print(f"local_energy_source: {args.local_energy_source}")
    print(f"optimizer: {args.optimizer_name}")
    print("gradient_mode: paper_tangent")
    print(f"checkpoint_roundtrip_bytes: {checkpoint_size}")
    print(f"initial_penalty_objective: {float(result.initial_terms['penalty_objective']):.12g}")
    print(f"final_penalty_objective: {float(result.final_terms['penalty_objective']):.12g}")
    for item in result.history:
        print(
            f"iteration_{item.iteration}: "
            f"acceptance={float(item.sampler.acceptance):.12g} "
            f"objective={float(item.update.penalty_objective):.12g} "
            f"grad_l2_norm={float(item.update.grad_l2_norm):.12g} "
            f"param_delta_l2_norm={float(item.update.param_delta_l2_norm):.12g} "
            f"optimizer_update_l2_norm={float(item.update.optimizer_update_l2_norm):.12g} "
            f"share_projection_l2_norm={float(item.update.share_projection_l2_norm):.12g} "
            f"update_accepted={bool(item.update.update_accepted)}"
        )
    if output_dir is not None:
        print(f"summary_json: {summary['summary_json']}")
        print(f"summary_markdown: {summary['summary_markdown']}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 2:
        raise ValueError("--walkers must be at least 2 for a nondegenerate driver smoke")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.burn_in < 0:
        raise ValueError("--burn-in must be non-negative")
    if args.sampler_steps < 1:
        raise ValueError("--sampler-steps must be at least 1")
    if args.proposal_width <= 0.0:
        raise ValueError("--proposal-width must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.optimizer_name not in {"sgd", "adam", "lamb", "kfac"}:
        raise ValueError(
            "--optimizer must be one of: sgd, adam, lamb, kfac; "
            f"got {args.optimizer_name!r}"
        )
    if args.candidate_check_period < 1:
        raise ValueError("--candidate-check-period must be at least 1")
    if args.kfac_damping is not None and args.kfac_damping <= 0.0:
        raise ValueError("--kfac-damping must be positive when set")
    if args.kfac_momentum is not None and (
        args.kfac_momentum < 0.0 or args.kfac_momentum >= 1.0
    ):
        raise ValueError("--kfac-momentum must be in [0, 1) when set")
    if args.kfac_norm_constraint is not None and args.kfac_norm_constraint <= 0.0:
        raise ValueError("--kfac-norm-constraint must be positive when set")
    if args.kfac_l2_reg is not None and args.kfac_l2_reg < 0.0:
        raise ValueError("--kfac-l2-reg must be non-negative when set")
    if args.kfac_cov_ema_decay is not None and (
        args.kfac_cov_ema_decay < 0.0 or args.kfac_cov_ema_decay >= 1.0
    ):
        raise ValueError("--kfac-cov-ema-decay must be in [0, 1) when set")
    if args.kfac_invert_every is not None and args.kfac_invert_every < 1:
        raise ValueError("--kfac-invert-every must be at least 1 when set")
    if args.overlap_ewma_decay is not None and (
        args.overlap_ewma_decay < 0.0 or args.overlap_ewma_decay >= 1.0
    ):
        raise ValueError("--overlap-ewma-decay must be in [0, 1)")
    if args.local_energy_source not in {"cheap", "real_pbc"}:
        raise ValueError(
            "--local-energy-source must be one of: cheap, real_pbc; "
            f"got {args.local_energy_source!r}"
        )


def _apply_runtime_defaults(experiment_path: Path, *, platform: str | None) -> None:
    from solidnes.backends.deepsolid_adapter import load_yaml

    runtime = load_yaml(experiment_path).get("runtime", {})
    if bool(runtime.get("x64_enabled", False)):
        os.environ["JAX_ENABLE_X64"] = "1"
    else:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        str(runtime.get("xla_python_client_preallocate", True)).lower(),
    )
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        str(runtime.get("xla_python_client_mem_fraction", 0.90)),
    )
    if platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)


def _resolve_runtime_args(args: argparse.Namespace, experiment: dict[str, Any]) -> None:
    diagnostics = experiment.get("diagnostics", {})
    sampler = diagnostics.get("sampler", {})
    optimizer = diagnostics.get("optimizer", {})
    update_guards = diagnostics.get("update_guards", {})
    overlap_scaling = diagnostics.get("overlap_scaling", {})
    parameter_sharing = diagnostics.get("parameter_sharing", {})
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
        default=2,
    )
    args.iterations = _resolve_int_option(
        args.iterations,
        env_name="SOLIDNES_NES_ITERATIONS",
        config_value=diagnostics.get("iterations"),
        default=2,
    )
    args.burn_in = _resolve_int_option(
        args.burn_in,
        env_name="SOLIDNES_NES_BURN_IN",
        config_value=sampler.get("burn_in"),
        default=1,
    )
    args.sampler_steps = _resolve_int_option(
        args.sampler_steps,
        env_name="SOLIDNES_NES_SAMPLER_STEPS",
        config_value=sampler.get("steps_per_iteration"),
        default=2,
    )
    args.seed = _resolve_int_option(
        args.seed,
        env_name="SOLIDNES_NES_SEED",
        config_value=diagnostics.get("seed"),
        default=53,
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
        config_value=optimizer.get("learning_rate", diagnostics.get("learning_rate")),
        default=1.0e-8,
    )
    args.optimizer_name = _resolve_str_option(
        args.optimizer_name,
        env_name="SOLIDNES_NES_OPTIMIZER",
        config_value=optimizer.get("name"),
        default="sgd",
    ).lower()
    args.adam_b1 = _resolve_float_option(
        args.adam_b1,
        env_name="SOLIDNES_NES_ADAM_B1",
        config_value=optimizer.get("adam_b1"),
        default=0.9,
    )
    args.adam_b2 = _resolve_float_option(
        args.adam_b2,
        env_name="SOLIDNES_NES_ADAM_B2",
        config_value=optimizer.get("adam_b2"),
        default=0.999,
    )
    args.adam_eps = _resolve_float_option(
        args.adam_eps,
        env_name="SOLIDNES_NES_ADAM_EPS",
        config_value=optimizer.get("adam_eps"),
        default=1.0e-8,
    )
    args.weight_decay = _resolve_float_option(
        args.weight_decay,
        env_name="SOLIDNES_NES_WEIGHT_DECAY",
        config_value=optimizer.get("weight_decay"),
        default=0.0,
    )
    args.lamb_eps = _resolve_float_option(
        args.lamb_eps,
        env_name="SOLIDNES_NES_LAMB_EPS",
        config_value=optimizer.get("lamb_eps"),
        default=1.0e-6,
    )
    kfac = optimizer.get("kfac", {})
    args.kfac_damping = _resolve_optional_float_option(
        args.kfac_damping,
        env_name="SOLIDNES_NES_KFAC_DAMPING",
        config_value=optimizer.get("kfac_damping", kfac.get("damping")),
    )
    args.kfac_momentum = _resolve_optional_float_option(
        args.kfac_momentum,
        env_name="SOLIDNES_NES_KFAC_MOMENTUM",
        config_value=optimizer.get("kfac_momentum", kfac.get("momentum")),
    )
    args.kfac_norm_constraint = _resolve_optional_float_option(
        args.kfac_norm_constraint,
        env_name="SOLIDNES_NES_KFAC_NORM_CONSTRAINT",
        config_value=optimizer.get(
            "kfac_norm_constraint",
            kfac.get("norm_constraint"),
        ),
    )
    args.kfac_l2_reg = _resolve_optional_float_option(
        args.kfac_l2_reg,
        env_name="SOLIDNES_NES_KFAC_L2_REG",
        config_value=optimizer.get("kfac_l2_reg", kfac.get("l2_reg")),
    )
    args.kfac_cov_ema_decay = _resolve_optional_float_option(
        args.kfac_cov_ema_decay,
        env_name="SOLIDNES_NES_KFAC_COV_EMA_DECAY",
        config_value=optimizer.get(
            "kfac_cov_ema_decay",
            kfac.get("cov_ema_decay"),
        ),
    )
    args.kfac_invert_every = _resolve_int_option(
        args.kfac_invert_every,
        env_name="SOLIDNES_NES_KFAC_INVERT_EVERY",
        config_value=optimizer.get("kfac_invert_every", kfac.get("invert_every")),
        default=None,
    )
    args.kfac_register_only_generic = bool(
        args.kfac_register_only_generic
        or optimizer.get(
            "kfac_register_only_generic",
            kfac.get("register_only_generic", False),
        )
    )
    args.max_grad_l2_norm = _resolve_optional_float_option(
        args.max_grad_l2_norm,
        env_name="SOLIDNES_NES_MAX_GRAD_L2_NORM",
        config_value=update_guards.get("max_grad_l2_norm"),
    )
    args.max_update_l2_norm = _resolve_optional_float_option(
        args.max_update_l2_norm,
        env_name="SOLIDNES_NES_MAX_UPDATE_L2_NORM",
        config_value=update_guards.get("max_update_l2_norm"),
    )
    args.overlap_ewma_decay = _resolve_optional_float_option(
        args.overlap_ewma_decay,
        env_name="SOLIDNES_NES_OVERLAP_EWMA_DECAY",
        config_value=overlap_scaling.get("ewma_decay"),
    )
    args.param_share_keys = _resolve_merge_keys(
        args.param_share_keys,
        env_name="SOLIDNES_NES_PARAM_SHARE_KEYS",
        config_value=parameter_sharing.get("merge_keys"),
    )
    args.candidate_check_period = _resolve_int_option(
        args.candidate_check_period,
        env_name="SOLIDNES_NES_CANDIDATE_CHECK_PERIOD",
        config_value=update_guards.get("candidate_check_period"),
        default=1,
    )
    args.proposal_width = _resolve_float_option(
        args.proposal_width,
        env_name="SOLIDNES_NES_PROPOSAL_WIDTH",
        config_value=sampler.get("proposal_width"),
        default=0.02,
    )
    args.local_energy_source = _resolve_str_option(
        args.local_energy_source,
        env_name="SOLIDNES_NES_LOCAL_ENERGY_SOURCE",
        config_value=diagnostics.get("local_energy_source"),
        default="cheap",
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


def _resolve_optional_float_option(
    value: float | None,
    *,
    env_name: str,
    config_value: Any,
) -> float | None:
    if value is not None:
        return float(value)
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        return float(env_value)
    if config_value is not None:
        return float(config_value)
    return None


def _resolve_str_option(
    value: str | None,
    *,
    env_name: str,
    config_value: Any,
    default: str,
) -> str:
    if value is not None:
        return str(value)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return str(env_value)
    if config_value is not None:
        return str(config_value)
    return default


def _resolve_merge_keys(
    value: str | None,
    *,
    env_name: str,
    config_value: Any,
) -> tuple[str, ...]:
    if value is not None:
        return _split_merge_keys(value)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return _split_merge_keys(env_value)
    if config_value is None:
        return ()
    if isinstance(config_value, str):
        return _split_merge_keys(config_value)
    return tuple(str(item).strip() for item in config_value if str(item).strip())


def _split_merge_keys(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _cheap_local_energy(adapter):
    jnp = adapter.modules.jnp

    def local_energy(params, positions, spins, atoms, charges):
        del params, atoms, charges
        position_scale = jnp.mean(positions * positions, axis=-1)
        spin_scale = 0.01 * jnp.sum(spins, axis=-1)
        return position_scale + spin_scale

    return local_energy


def _validate_result(adapter: Any, result: Any, args: argparse.Namespace) -> None:
    expected_shape = (args.states, args.walkers)
    observed = tuple(adapter.modules.jnp.asarray(result.samples.positions).shape[:2])
    if observed != expected_shape:
        raise ValueError(f"sample shape mismatch: expected {expected_shape}, got {observed}")
    _assert_all_finite(adapter, "initial_penalty_objective", result.initial_terms["penalty_objective"])
    _assert_all_finite(adapter, "final_penalty_objective", result.final_terms["penalty_objective"])
    if len(result.history) != args.iterations:
        raise ValueError(f"history length mismatch: {len(result.history)} != {args.iterations}")
    for item in result.history:
        _assert_all_finite(adapter, f"acceptance_{item.iteration}", item.sampler.acceptance)
        acceptance = float(item.sampler.acceptance)
        if acceptance < 0.0 or acceptance > 1.0:
            raise ValueError(f"acceptance out of range: {acceptance}")
        _assert_positive_finite(
            adapter,
            f"grad_l2_norm_{item.iteration}",
            item.update.grad_l2_norm,
        )
        _assert_positive_finite(
            adapter,
            f"param_delta_l2_norm_{item.iteration}",
            item.update.param_delta_l2_norm,
        )
        _assert_positive_finite(
            adapter,
            f"optimizer_update_l2_norm_{item.iteration}",
            item.update.optimizer_update_l2_norm,
        )
        _assert_all_finite(
            adapter,
            f"share_projection_l2_norm_{item.iteration}",
            item.update.share_projection_l2_norm,
        )
        if args.max_update_l2_norm is not None:
            update_norm = float(item.update.optimizer_update_l2_norm)
            if update_norm > args.max_update_l2_norm * (1.0 + 1.0e-5):
                raise ValueError(
                    f"optimizer_update_l2_norm_{item.iteration} exceeds cap: "
                    f"{update_norm} > {args.max_update_l2_norm}"
                )
        _assert_bool_true(
            f"gradient_objective_finite_{item.iteration}",
            item.update.gradient_objective_finite,
        )
        _assert_bool_true(f"gradient_finite_{item.iteration}", item.update.gradient_finite)
        _assert_bool_true(f"update_finite_{item.iteration}", item.update.update_finite)
        _assert_bool_true(
            f"candidate_terms_finite_{item.iteration}",
            item.update.candidate_terms_finite,
        )
        _assert_bool_true(f"update_accepted_{item.iteration}", item.update.update_accepted)


def _roundtrip_checkpoint(adapter: Any, result: Any, metadata: dict[str, Any]) -> int:
    from solidnes.excited_states.ferminet_pbc_driver import (
        load_external_state_driver_checkpoint,
        save_external_state_driver_checkpoint,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "driver_checkpoint.pkl"
        save_external_state_driver_checkpoint(
            checkpoint,
            result=result,
            metadata=metadata,
        )
        loaded = load_external_state_driver_checkpoint(checkpoint, adapter=adapter)
        if loaded["metadata"] != metadata:
            raise ValueError("checkpoint metadata roundtrip mismatch")
        if result.optimizer_state is not None:
            if loaded.get("optimizer_state") is None:
                raise ValueError("checkpoint optimizer_state missing after roundtrip")
            if loaded["optimizer_state"].name != result.optimizer_state.name:
                raise ValueError("checkpoint optimizer_state name mismatch")
        if result.running_stats is not None and loaded.get("running_stats") is None:
            raise ValueError("checkpoint running_stats missing after roundtrip")
        loaded_shape = tuple(adapter.modules.jnp.asarray(loaded["samples"].positions).shape)
        result_shape = tuple(adapter.modules.jnp.asarray(result.samples.positions).shape)
        if loaded_shape != result_shape:
            raise ValueError(
                f"checkpoint sample shape mismatch: {loaded_shape} != {result_shape}"
            )
        return checkpoint.stat().st_size


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


def _assert_positive_finite(adapter: Any, name: str, value: Any) -> None:
    _assert_all_finite(adapter, name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _assert_bool_true(name: str, value: Any) -> None:
    if not bool(value):
        raise ValueError(f"{name} must be true, got {value}")


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "real": np.real(array).tolist(),
            "imag": np.imag(array).tolist(),
        }
    return array.tolist()


def _history_item(item: Any) -> dict[str, Any]:
    return {
        "iteration": int(item.iteration),
        "sampler_steps": int(item.sampler.steps),
        "sampler_acceptance": _json_value(item.sampler.acceptance),
        "penalty_objective": _json_value(item.update.penalty_objective),
        "gradient_objective": _json_value(item.update.gradient_objective),
        "state_energy": _json_value(item.update.state_energy),
        "offdiag_squared_overlap": _json_value(item.update.offdiag_squared_overlap),
        "grad_l2_norm": _json_value(item.update.grad_l2_norm),
        "param_delta_l2_norm": _json_value(item.update.param_delta_l2_norm),
        "optimizer_update_l2_norm": _json_value(item.update.optimizer_update_l2_norm),
        "share_projection_l2_norm": _json_value(item.update.share_projection_l2_norm),
        "optimizer_name": item.update.optimizer_name,
        "optimizer_step": item.update.optimizer_step,
        "candidate_check_performed": _json_value(
            item.update.candidate_check_performed
        ),
        "shared_param_paths": list(item.update.shared_param_paths),
        "gradient_finite": _json_value(item.update.gradient_finite),
        "update_finite": _json_value(item.update.update_finite),
        "candidate_terms_finite": _json_value(item.update.candidate_terms_finite),
        "update_accepted": _json_value(item.update.update_accepted),
    }


def _format_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# FermiNet PBC Driver Smoke",
        "",
        "```text",
        f"status: {summary['status']}",
        f"experiment: {summary['experiment']}",
        f"jax_platform: {summary['jax_platform']}",
        f"external_state_params: {summary['external_state_params']}",
        f"walkers_per_state: {summary['walkers_per_state']}",
        f"iterations: {summary['iterations']}",
        f"burn_in: {summary['burn_in']}",
        f"sampler_steps_per_iteration: {summary['sampler_steps_per_iteration']}",
        f"proposal_width: {summary['proposal_width']}",
        f"optimizer: {summary['optimizer']}",
        f"learning_rate: {summary['learning_rate']}",
        f"kfac_damping: {summary['kfac_damping']}",
        f"kfac_momentum: {summary['kfac_momentum']}",
        f"kfac_norm_constraint: {summary['kfac_norm_constraint']}",
        f"kfac_invert_every: {summary['kfac_invert_every']}",
        f"overlap_ewma_decay: {summary['overlap_ewma_decay']}",
        f"param_share_keys: {summary['param_share_keys']}",
        f"candidate_check_period: {summary['candidate_check_period']}",
        f"gradient_mode: {summary['gradient_mode']}",
        f"local_energy_source: {summary['local_energy_source']}",
        f"checkpoint_roundtrip_bytes: {summary['checkpoint_roundtrip_bytes']}",
        f"initial_penalty_objective: {summary['initial_penalty_objective']}",
        f"final_penalty_objective: {summary['final_penalty_objective']}",
        "```",
        "",
    ]
    for item in summary["history"]:
        lines.extend(
            [
                "```text",
                f"iteration: {item['iteration']}",
                f"sampler_steps: {item['sampler_steps']}",
                f"sampler_acceptance: {item['sampler_acceptance']}",
                f"penalty_objective: {item['penalty_objective']}",
                f"gradient_objective: {item['gradient_objective']}",
                f"state_energy: {item['state_energy']}",
                f"offdiag_squared_overlap: {item['offdiag_squared_overlap']}",
                f"optimizer_step: {item['optimizer_step']}",
                f"candidate_check_performed: {item['candidate_check_performed']}",
                f"shared_param_paths: {item['shared_param_paths']}",
                f"grad_l2_norm: {item['grad_l2_norm']}",
                f"param_delta_l2_norm: {item['param_delta_l2_norm']}",
                f"optimizer_update_l2_norm: {item['optimizer_update_l2_norm']}",
                f"share_projection_l2_norm: {item['share_projection_l2_norm']}",
                f"gradient_finite: {item['gradient_finite']}",
                f"update_finite: {item['update_finite']}",
                f"candidate_terms_finite: {item['candidate_terms_finite']}",
                f"update_accepted: {item['update_accepted']}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _resolve_output_dir(
    args: argparse.Namespace,
    experiment: dict[str, Any],
) -> Path | None:
    if args.output_dir:
        return _resolve_project_path(args.output_dir)
    task_root = os.environ.get("SOLIDNES_TASK_ROOT")
    if task_root:
        return _resolve_project_path(task_root) / "results" / "validation"
    validation_dir = experiment.get("output", {}).get("validation_dir")
    if validation_dir:
        return _resolve_project_path(validation_dir)
    return None


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
