#!/usr/bin/env python
"""Build-only multi-step optimization smoke for FermiNet PBC penalty states."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    "configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml"
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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--penalty-alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-8)
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=("sgd", "adam", "lamb", "kfac"),
    )
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lamb-eps", type=float, default=1.0e-6)
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
    parser.add_argument("--candidate-check-period", type=int, default=1)
    parser.add_argument(
        "--platform",
        default="cpu",
        help="JAX platform for this build-only smoke. Use '' to leave unset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 1:
        raise ValueError("--walkers must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
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
    if args.platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)
        os.environ.setdefault("JAX_PLATFORMS", args.platform)

    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        assert_pbc_external_state_config,
        build_external_state_adapter,
    )
    from solidnes.excited_states.ferminet_pbc_training import (
        run_external_state_penalty_sgd,
    )

    bundle = build_ferminet_adapter(args.experiment)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)
    cheap_local_energy, local_energy_calls = _counted_cheap_local_energy(adapter)

    jax = adapter.modules.jax
    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key = jax.random.split(key, 3)
    params = adapter.init_state_params(params_key, args.states)
    samples = adapter.tiny_state_samples(
        samples_key,
        nstates=args.states,
        walkers=args.walkers,
    )

    result = run_external_state_penalty_sgd(
        adapter,
        params,
        samples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        penalty_alpha=args.penalty_alpha,
        local_energy=cheap_local_energy,
        optimizer_name=args.optimizer,
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
        max_grad_l2_norm=args.max_grad_l2_norm,
        max_update_l2_norm=args.max_update_l2_norm,
        overlap_ewma_decay=args.overlap_ewma_decay,
        param_share_keys=_split_merge_keys(args.param_share_keys),
        candidate_check_period=args.candidate_check_period,
        block_until_ready=True,
    )
    initial_value = result.initial_terms["penalty_objective"]
    _assert_finite(adapter, "initial_penalty_objective", initial_value)
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
            f"grad_norm_step_{step}",
            diagnostics.grad_l2_norm,
        )
        _assert_positive_finite(
            adapter,
            f"delta_norm_step_{step}",
            diagnostics.param_delta_l2_norm,
        )
        _assert_positive_finite(
            adapter,
            f"optimizer_update_norm_step_{step}",
            diagnostics.optimizer_update_l2_norm,
        )
        _assert_finite(
            adapter,
            f"share_projection_norm_step_{step}",
            diagnostics.share_projection_l2_norm,
        )
        if args.max_update_l2_norm is not None:
            update_norm = float(diagnostics.optimizer_update_l2_norm)
            if update_norm > args.max_update_l2_norm * (1.0 + 1.0e-5):
                raise ValueError(
                    f"optimizer_update_norm_step_{step} exceeds cap: "
                    f"{update_norm} > {args.max_update_l2_norm}"
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

    final_value = result.final_terms["penalty_objective"]
    _assert_finite(adapter, "final_penalty_objective", final_value)
    candidate_checks = sum(
        1 for step in range(args.steps) if step % args.candidate_check_period == 0
    )
    expected_local_energy_calls = args.states * (2 + args.steps + candidate_checks)
    if local_energy_calls["count"] != expected_local_energy_calls:
        raise ValueError(
            "paper_tangent objective called local_energy inside value_and_grad: "
            f"expected {expected_local_energy_calls} calls, got "
            f"{local_energy_calls['count']}"
        )

    print("ferminet_pbc_penalty_opt_smoke: ok")
    print(f"experiment: {bundle.paths.experiment.relative_to(PROJECT_ROOT)}")
    print(f"jax_platform: {jax.default_backend()}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print("local_energy_source: cheap")
    print("gradient_mode: paper_tangent")
    print(f"optimizer: {args.optimizer}")
    print(f"overlap_ewma_decay: {args.overlap_ewma_decay}")
    print(f"param_share_keys: {_split_merge_keys(args.param_share_keys)}")
    print(f"candidate_check_period: {args.candidate_check_period}")
    print(f"local_energy_calls: {local_energy_calls['count']}")
    print(f"steps: {args.steps}")
    print(f"initial_penalty_objective: {float(initial_value):.12g}")
    print(f"final_penalty_objective: {float(final_value):.12g}")
    print(f"objective_delta: {float(final_value - initial_value):.12g}")
    for diagnostics in result.history:
        print(
            f"step_{diagnostics.step}: "
            f"objective={float(diagnostics.penalty_objective):.12g} "
            f"gradient_objective={float(diagnostics.gradient_objective):.12g} "
            f"grad_l2_norm={float(diagnostics.grad_l2_norm):.12g} "
            f"param_delta_l2_norm={float(diagnostics.param_delta_l2_norm):.12g} "
            f"optimizer_update_l2_norm={float(diagnostics.optimizer_update_l2_norm):.12g} "
            f"share_projection_l2_norm={float(diagnostics.share_projection_l2_norm):.12g} "
            f"candidate_check_performed={bool(diagnostics.candidate_check_performed)} "
            f"shared_param_paths={len(diagnostics.shared_param_paths)} "
            f"update_accepted={bool(diagnostics.update_accepted)}"
        )
    return 0


def _split_merge_keys(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _counted_cheap_local_energy(adapter):
    jnp = adapter.modules.jnp
    calls = {"count": 0}

    def local_energy(params, positions, spins, atoms, charges):
        calls["count"] += 1
        del params, atoms, charges
        position_scale = jnp.mean(positions * positions, axis=-1)
        spin_scale = 0.01 * jnp.sum(spins, axis=-1)
        return position_scale + spin_scale

    return local_energy, calls


def _assert_finite(adapter, name: str, value) -> None:
    if not bool(adapter.modules.jnp.isfinite(value)):
        raise ValueError(f"{name} is not finite: {value}")


def _assert_positive_finite(adapter, name: str, value) -> None:
    _assert_finite(adapter, name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _assert_bool_true(adapter, name: str, value) -> None:
    if not bool(value):
        raise ValueError(f"{name} must be true, got {value}")


if __name__ == "__main__":
    raise SystemExit(main())
