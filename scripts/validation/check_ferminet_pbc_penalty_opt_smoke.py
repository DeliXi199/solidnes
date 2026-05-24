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
    cheap_local_energy = _cheap_local_energy(adapter)

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

    final_value = result.final_terms["penalty_objective"]
    _assert_finite(adapter, "final_penalty_objective", final_value)
    if float(final_value) >= float(initial_value):
        raise ValueError(
            "final objective did not decrease: "
            f"{float(initial_value)} -> {float(final_value)}"
        )

    print("ferminet_pbc_penalty_opt_smoke: ok")
    print(f"experiment: {bundle.paths.experiment.relative_to(PROJECT_ROOT)}")
    print(f"jax_platform: {jax.default_backend()}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {args.states}")
    print("local_energy_source: cheap")
    print(f"steps: {args.steps}")
    print(f"initial_penalty_objective: {float(initial_value):.12g}")
    print(f"final_penalty_objective: {float(final_value):.12g}")
    print(f"objective_delta: {float(final_value - initial_value):.12g}")
    for diagnostics in result.history:
        print(
            f"step_{diagnostics.step}: "
            f"objective={float(diagnostics.penalty_objective):.12g} "
            f"grad_l2_norm={float(diagnostics.grad_l2_norm):.12g} "
            f"param_delta_l2_norm={float(diagnostics.param_delta_l2_norm):.12g}"
        )
    return 0


def _cheap_local_energy(adapter):
    jnp = adapter.modules.jnp

    def local_energy(params, positions, spins, atoms, charges):
        del params, atoms, charges
        position_scale = jnp.mean(positions * positions, axis=-1)
        spin_scale = 0.01 * jnp.sum(spins, axis=-1)
        return position_scale + spin_scale

    return local_energy


def _assert_finite(adapter, name: str, value) -> None:
    if not bool(adapter.modules.jnp.isfinite(value)):
        raise ValueError(f"{name} is not finite: {value}")


def _assert_positive_finite(adapter, name: str, value) -> None:
    _assert_finite(adapter, name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


if __name__ == "__main__":
    raise SystemExit(main())
