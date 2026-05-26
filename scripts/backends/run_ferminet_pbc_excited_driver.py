#!/usr/bin/env python
"""Run the FermiNet PBC external-state excited-state driver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (SRC_ROOT, FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_EXPERIMENT = (
    "configs/experiment/"
    "diamond_c_ferminet_pbc_gamma_driver_real_local_energy_walkers4_smoke.yaml"
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
    parser.add_argument(
        "--param-share-keys",
        default=None,
        help="Comma-separated parameter path substrings to average across states.",
    )
    parser.add_argument("--candidate-check-period", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional driver checkpoint to resume from.",
    )
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
        help="Optional directory for JSON/Markdown run summaries.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for driver checkpoints.",
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
        evaluate_ferminet_pbc_penalty_terms,
    )
    from solidnes.excited_states.ferminet_pbc_driver import (
        load_external_state_driver_checkpoint,
        run_external_state_penalty_driver,
        save_external_state_driver_checkpoint,
    )

    experiment = load_yaml(experiment_path)
    _resolve_runtime_args(args, experiment)
    _validate_args(args)

    bundle = build_ferminet_adapter(experiment_path)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)
    local_energy = _cheap_local_energy(adapter) if args.local_energy_source == "cheap" else None
    jax = adapter.modules.jax

    checkpoint_dir = _resolve_checkpoint_dir(args, experiment)
    output_dir = _resolve_output_dir(args, experiment)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_checkpoint:
        resume_path = _resolve_project_path(args.resume_checkpoint)
        payload = load_external_state_driver_checkpoint(resume_path, adapter=adapter)
        params = payload["state_params"]
        samples = payload["samples"]
        sampler_key = payload["sampler_key"]
        optimizer_state = payload.get("optimizer_state")
        running_stats = payload.get("running_stats")
        completed_iterations = int(
            payload.get("metadata", {}).get("completed_iterations", 0)
        )
        resume_display = _display_path(resume_path)
    else:
        key = jax.random.PRNGKey(args.seed)
        _, params_key, samples_key, sampler_key = jax.random.split(key, 4)
        params = adapter.init_state_params(params_key, args.states)
        samples = adapter.tiny_state_samples(
            samples_key,
            nstates=args.states,
            walkers=args.walkers,
        )
        optimizer_state = None
        running_stats = None
        completed_iterations = 0
        resume_display = None

    _validate_sample_shape(adapter, samples, args)
    first_terms = None
    final_terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        samples,
        penalty_alpha=args.penalty_alpha,
        local_energy=local_energy,
    )
    history = []
    checkpoint_paths = []
    start = time.perf_counter()
    while completed_iterations < args.iterations:
        segment_start = completed_iterations
        segment_iterations = min(
            args.checkpoint_interval,
            args.iterations - completed_iterations,
        )
        burn_in = args.burn_in if completed_iterations == 0 else 0
        segment = run_external_state_penalty_driver(
            adapter,
            params,
            samples,
            sampler_key=sampler_key,
            iterations=segment_iterations,
            learning_rate=args.learning_rate,
            penalty_alpha=args.penalty_alpha,
            sampler_burn_in=burn_in,
            sampler_steps_per_iteration=args.sampler_steps,
            sampler_proposal_width=args.proposal_width,
            local_energy=local_energy,
            optimizer_name=args.optimizer_name,
            optimizer_state=optimizer_state,
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
            running_stats=running_stats,
            overlap_ewma_decay=args.overlap_ewma_decay,
            max_grad_l2_norm=args.max_grad_l2_norm,
            max_update_l2_norm=args.max_update_l2_norm,
            param_share_keys=args.param_share_keys,
            candidate_check_period=args.candidate_check_period,
            block_until_ready=True,
        )
        if first_terms is None:
            first_terms = segment.initial_terms
        params = segment.state_params
        samples = segment.samples
        sampler_key = segment.sampler_key
        optimizer_state = segment.optimizer_state
        running_stats = segment.running_stats
        final_terms = segment.final_terms
        for item in segment.history:
            history.append(
                _history_item(
                    item,
                    absolute_iteration=segment_start + int(item.iteration),
                )
            )
        completed_iterations += segment_iterations
        checkpoint_path = checkpoint_dir / f"driver_iter_{completed_iterations:06d}.pkl"
        save_external_state_driver_checkpoint(
            checkpoint_path,
            result=segment,
            metadata={
                "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
                "completed_iterations": completed_iterations,
                "target_iterations": args.iterations,
                "states": args.states,
                "walkers": args.walkers,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer_name,
                "adam_b1": args.adam_b1,
                "adam_b2": args.adam_b2,
                "adam_eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "lamb_eps": args.lamb_eps,
                "kfac_damping": args.kfac_damping,
                "kfac_momentum": args.kfac_momentum,
                "kfac_norm_constraint": args.kfac_norm_constraint,
                "kfac_l2_reg": args.kfac_l2_reg,
                "kfac_cov_ema_decay": args.kfac_cov_ema_decay,
                "kfac_invert_every": args.kfac_invert_every,
                "kfac_register_only_generic": args.kfac_register_only_generic,
                "optimizer_state_metadata": None
                if optimizer_state is None
                else optimizer_state.native_state_metadata,
                "overlap_ewma_decay": args.overlap_ewma_decay,
                "param_share_keys": list(args.param_share_keys),
                "candidate_check_period": args.candidate_check_period,
                "penalty_alpha": args.penalty_alpha,
                "local_energy_source": args.local_energy_source,
            },
        )
        checkpoint_paths.append(_display_path(checkpoint_path))

    elapsed_seconds = time.perf_counter() - start
    if first_terms is None:
        first_terms = final_terms
    _validate_terms(adapter, "initial", first_terms, args)
    _validate_terms(adapter, "final", final_terms, args)
    _validate_history(
        adapter,
        history,
        max_update_l2_norm=args.max_update_l2_norm,
    )

    summary = {
        "status": "ok",
        "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cfg_system_states": cfg_states,
        "external_state_params": args.states,
        "walkers_per_state": args.walkers,
        "target_iterations": args.iterations,
        "completed_iterations": completed_iterations,
        "checkpoint_interval": args.checkpoint_interval,
        "burn_in": args.burn_in,
        "sampler_steps_per_iteration": args.sampler_steps,
        "proposal_width": args.proposal_width,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer_name,
        "adam_b1": args.adam_b1,
        "adam_b2": args.adam_b2,
        "adam_eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "lamb_eps": args.lamb_eps,
        "kfac_damping": args.kfac_damping,
        "kfac_momentum": args.kfac_momentum,
        "kfac_norm_constraint": args.kfac_norm_constraint,
        "kfac_l2_reg": args.kfac_l2_reg,
        "kfac_cov_ema_decay": args.kfac_cov_ema_decay,
        "kfac_invert_every": args.kfac_invert_every,
        "kfac_register_only_generic": args.kfac_register_only_generic,
        "optimizer_state_metadata": None
        if optimizer_state is None
        else optimizer_state.native_state_metadata,
        "max_grad_l2_norm": args.max_grad_l2_norm,
        "max_update_l2_norm": args.max_update_l2_norm,
        "overlap_ewma_decay": args.overlap_ewma_decay,
        "param_share_keys": list(args.param_share_keys),
        "candidate_check_period": args.candidate_check_period,
        "penalty_alpha": args.penalty_alpha,
        "gradient_mode": "paper_tangent",
        "local_energy_source": args.local_energy_source,
        "resume_checkpoint": resume_display,
        "elapsed_seconds": elapsed_seconds,
        "checkpoints": checkpoint_paths,
        "initial_penalty_objective": _json_value(first_terms["penalty_objective"]),
        "final_penalty_objective": _json_value(final_terms["penalty_objective"]),
        "initial_state_energy": _json_value(first_terms["state_energy"]),
        "final_state_energy": _json_value(final_terms["state_energy"]),
        "final_overlap_matrix": _json_value(final_terms["overlap_matrix"]),
        "history": history,
    }
    if output_dir is not None:
        json_path = output_dir / "ferminet_pbc_driver_run_summary.json"
        md_path = output_dir / "ferminet_pbc_driver_run_summary.md"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown_summary(summary), encoding="utf-8")
        summary["summary_json"] = _display_path(json_path)
        summary["summary_markdown"] = _display_path(md_path)

    print("ferminet_pbc_excited_driver_run: ok")
    print(f"experiment: {summary['experiment']}")
    print(f"jax_platform: {summary['jax_platform']}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print(f"walkers_per_state: {args.walkers}")
    print(f"completed_iterations: {completed_iterations}")
    print(f"checkpoint_interval: {args.checkpoint_interval}")
    print(f"optimizer: {args.optimizer_name}")
    print(f"local_energy_source: {args.local_energy_source}")
    print(f"elapsed_seconds: {elapsed_seconds:.6f}")
    print(f"initial_penalty_objective: {float(first_terms['penalty_objective']):.12g}")
    print(f"final_penalty_objective: {float(final_terms['penalty_objective']):.12g}")
    for item in history:
        print(
            f"iteration_{item['iteration']}: "
            f"acceptance={float(item['sampler_acceptance']):.12g} "
            f"objective={float(item['penalty_objective']):.12g} "
            f"grad_l2_norm={float(item['grad_l2_norm']):.12g} "
            f"param_delta_l2_norm={float(item['param_delta_l2_norm']):.12g} "
            f"optimizer_update_l2_norm={float(item['optimizer_update_l2_norm']):.12g} "
            f"share_projection_l2_norm={float(item['share_projection_l2_norm']):.12g} "
            f"update_accepted={bool(item['update_accepted'])}"
        )
    for checkpoint_path in checkpoint_paths:
        print(f"checkpoint: {checkpoint_path}")
    if output_dir is not None:
        print(f"summary_json: {summary['summary_json']}")
        print(f"summary_markdown: {summary['summary_markdown']}")
    return 0


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
    checkpoint = diagnostics.get("checkpoint", {})
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
        default=4,
    )
    args.iterations = _resolve_int_option(
        args.iterations,
        env_name="SOLIDNES_NES_ITERATIONS",
        config_value=diagnostics.get("iterations"),
        default=12,
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
        default=61,
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
    args.checkpoint_interval = _resolve_int_option(
        args.checkpoint_interval,
        env_name="SOLIDNES_NES_CHECKPOINT_INTERVAL",
        config_value=checkpoint.get("interval"),
        default=args.iterations,
    )
    args.local_energy_source = _resolve_str_option(
        args.local_energy_source,
        env_name="SOLIDNES_NES_LOCAL_ENERGY_SOURCE",
        config_value=diagnostics.get("local_energy_source"),
        default="real_pbc",
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 2:
        raise ValueError("--walkers must be at least 2")
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
    if args.adam_b1 < 0.0 or args.adam_b1 >= 1.0:
        raise ValueError("--adam-b1 must be in [0, 1)")
    if args.adam_b2 < 0.0 or args.adam_b2 >= 1.0:
        raise ValueError("--adam-b2 must be in [0, 1)")
    if args.adam_eps <= 0.0:
        raise ValueError("--adam-eps must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.lamb_eps <= 0.0:
        raise ValueError("--lamb-eps must be positive")
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
    if args.max_grad_l2_norm is not None and args.max_grad_l2_norm <= 0.0:
        raise ValueError("--max-grad-l2-norm must be positive when set")
    if args.max_update_l2_norm is not None and args.max_update_l2_norm <= 0.0:
        raise ValueError("--max-update-l2-norm must be positive when set")
    if args.overlap_ewma_decay is not None and (
        args.overlap_ewma_decay < 0.0 or args.overlap_ewma_decay >= 1.0
    ):
        raise ValueError("--overlap-ewma-decay must be in [0, 1)")
    if args.candidate_check_period < 1:
        raise ValueError("--candidate-check-period must be at least 1")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1")
    if args.local_energy_source not in {"cheap", "real_pbc"}:
        raise ValueError(
            "--local-energy-source must be one of: cheap, real_pbc; "
            f"got {args.local_energy_source!r}"
        )


def _cheap_local_energy(adapter):
    jnp = adapter.modules.jnp

    def local_energy(params, positions, spins, atoms, charges):
        del params, atoms, charges
        position_scale = jnp.mean(positions * positions, axis=-1)
        spin_scale = 0.01 * jnp.sum(spins, axis=-1)
        return position_scale + spin_scale

    return local_energy


def _validate_sample_shape(
    adapter: Any,
    samples: Any,
    args: argparse.Namespace,
) -> None:
    observed = tuple(adapter.modules.jnp.asarray(samples.positions).shape[:2])
    expected = (args.states, args.walkers)
    if observed != expected:
        raise ValueError(f"sample shape mismatch: expected {expected}, got {observed}")


def _validate_terms(
    adapter: Any,
    label: str,
    terms: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    expected_shapes = {
        "local_energy": (args.states, args.walkers),
        "state_energy": (args.states,),
        "overlap_matrix": (args.states, args.states),
        "penalty_objective": (),
    }
    for name, shape in expected_shapes.items():
        observed = tuple(adapter.modules.jnp.asarray(terms[name]).shape)
        if observed != shape:
            raise ValueError(
                f"{label}_{name} shape mismatch: expected {shape}, got {observed}"
            )
        _assert_all_finite(adapter, f"{label}_{name}", terms[name])


def _validate_history(
    adapter: Any,
    history: list[dict[str, Any]],
    *,
    max_update_l2_norm: float | None,
) -> None:
    if not history:
        return
    for item in history:
        acceptance = float(item["sampler_acceptance"])
        if acceptance < 0.0 or acceptance > 1.0:
            raise ValueError(f"acceptance out of range: {acceptance}")
        _assert_positive_finite(adapter, f"grad_l2_norm_{item['iteration']}", item["grad_l2_norm"])
        _assert_positive_finite(
            adapter,
            f"param_delta_l2_norm_{item['iteration']}",
            item["param_delta_l2_norm"],
        )
        _assert_positive_finite(
            adapter,
            f"optimizer_update_l2_norm_{item['iteration']}",
            item["optimizer_update_l2_norm"],
        )
        if max_update_l2_norm is not None:
            update_norm = float(item["optimizer_update_l2_norm"])
            if update_norm > max_update_l2_norm * (1.0 + 1.0e-5):
                raise ValueError(
                    f"optimizer_update_l2_norm_{item['iteration']} exceeds cap: "
                    f"{update_norm} > {max_update_l2_norm}"
                )
        _assert_all_finite(
            adapter,
            f"share_projection_l2_norm_{item['iteration']}",
            item["share_projection_l2_norm"],
        )
        for name in (
            "gradient_objective_finite",
            "gradient_finite",
            "update_finite",
            "candidate_terms_finite",
            "update_accepted",
        ):
            if not bool(item[name]):
                raise ValueError(f"{name}_{item['iteration']} must be true")


def _history_item(item: Any, *, absolute_iteration: int) -> dict[str, Any]:
    return {
        "iteration": int(absolute_iteration),
        "sampler_steps": int(item.sampler.steps),
        "sampler_acceptance": _json_value(item.sampler.acceptance),
        "penalty_objective": _json_value(item.update.penalty_objective),
        "gradient_objective": _json_value(item.update.gradient_objective),
        "state_energy": _json_value(item.update.state_energy),
        "state_energy_std": _json_value(item.update.state_energy_std),
        "state_energy_ewma": _json_value(item.update.state_energy_ewma),
        "state_energy_std_ewma": _json_value(item.update.state_energy_std_ewma),
        "overlap_gradient_scale": _json_value(item.update.overlap_gradient_scale),
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
        "gradient_objective_finite": _json_value(
            item.update.gradient_objective_finite
        ),
        "gradient_finite": _json_value(item.update.gradient_finite),
        "update_finite": _json_value(item.update.update_finite),
        "candidate_terms_finite": _json_value(item.update.candidate_terms_finite),
        "update_accepted": _json_value(item.update.update_accepted),
    }


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


def _resolve_checkpoint_dir(args: argparse.Namespace, experiment: dict[str, Any]) -> Path:
    if args.checkpoint_dir:
        return _resolve_project_path(args.checkpoint_dir)
    task_root = os.environ.get("SOLIDNES_TASK_ROOT")
    if task_root:
        return _resolve_project_path(task_root) / "results" / "checkpoints"
    checkpoint_dir = experiment.get("output", {}).get("checkpoint_dir")
    if checkpoint_dir:
        return _resolve_project_path(checkpoint_dir)
    return _resolve_project_path("outputs/checkpoints/ferminet_pbc_driver")


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
        "# FermiNet PBC Excited-State Driver Run",
        "",
        "```text",
        f"status: {summary['status']}",
        f"experiment: {summary['experiment']}",
        f"jax_platform: {summary['jax_platform']}",
        f"external_state_params: {summary['external_state_params']}",
        f"walkers_per_state: {summary['walkers_per_state']}",
        f"completed_iterations: {summary['completed_iterations']}",
        f"checkpoint_interval: {summary['checkpoint_interval']}",
        f"optimizer: {summary['optimizer']}",
        f"learning_rate: {summary['learning_rate']}",
        f"kfac_damping: {summary['kfac_damping']}",
        f"kfac_momentum: {summary['kfac_momentum']}",
        f"kfac_norm_constraint: {summary['kfac_norm_constraint']}",
        f"kfac_invert_every: {summary['kfac_invert_every']}",
        f"overlap_ewma_decay: {summary['overlap_ewma_decay']}",
        f"param_share_keys: {summary['param_share_keys']}",
        f"candidate_check_period: {summary['candidate_check_period']}",
        f"local_energy_source: {summary['local_energy_source']}",
        f"elapsed_seconds: {summary['elapsed_seconds']:.6f}",
        f"initial_penalty_objective: {summary['initial_penalty_objective']}",
        f"final_penalty_objective: {summary['final_penalty_objective']}",
        "```",
        "",
        "Checkpoints:",
        "",
    ]
    for checkpoint in summary["checkpoints"]:
        lines.append(f"- `{checkpoint}`")
    lines.extend(["", "Per-iteration diagnostics:", ""])
    for item in summary["history"]:
        lines.extend(
            [
                "```text",
                f"iteration: {item['iteration']}",
                f"sampler_steps: {item['sampler_steps']}",
                f"sampler_acceptance: {item['sampler_acceptance']}",
                f"penalty_objective: {item['penalty_objective']}",
                f"state_energy: {item['state_energy']}",
                f"state_energy_std: {item['state_energy_std']}",
                f"overlap_gradient_scale: {item['overlap_gradient_scale']}",
                f"offdiag_squared_overlap: {item['offdiag_squared_overlap']}",
                f"optimizer_step: {item['optimizer_step']}",
                f"candidate_check_performed: {item['candidate_check_performed']}",
                f"shared_param_paths: {item['shared_param_paths']}",
                f"grad_l2_norm: {item['grad_l2_norm']}",
                f"param_delta_l2_norm: {item['param_delta_l2_norm']}",
                f"optimizer_update_l2_norm: {item['optimizer_update_l2_norm']}",
                f"share_projection_l2_norm: {item['share_projection_l2_norm']}",
                f"update_accepted: {item['update_accepted']}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


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
