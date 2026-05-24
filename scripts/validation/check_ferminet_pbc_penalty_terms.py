#!/usr/bin/env python
"""Build-only FermiNet PBC penalty-term validation for external states."""

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
    parser.add_argument("--walkers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--penalty-alpha", type=float, default=1.0)
    parser.add_argument(
        "--platform",
        default="cpu",
        help="JAX platform for this build-only check. Use '' to leave unset.",
    )
    parser.add_argument(
        "--evaluate-local-energy",
        action="store_true",
        help="Use real PBC local energy instead of a cheap build-only stand-in.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.states < 2:
        raise ValueError("--states must be at least 2")
    if args.walkers < 1:
        raise ValueError("--walkers must be at least 1")
    if args.platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)
        os.environ.setdefault("JAX_PLATFORMS", args.platform)

    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        assert_pbc_external_state_config,
        build_external_state_adapter,
        evaluate_ferminet_pbc_penalty_terms,
    )

    bundle = build_ferminet_adapter(args.experiment)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)

    jax = adapter.modules.jax
    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key = jax.random.split(key, 3)
    state_params = adapter.init_state_params(params_key, args.states)
    samples = adapter.tiny_state_samples(
        samples_key,
        nstates=args.states,
        walkers=args.walkers,
    )
    local_energy = None if args.evaluate_local_energy else _cheap_local_energy(adapter)
    terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        state_params,
        samples,
        penalty_alpha=args.penalty_alpha,
        local_energy=local_energy,
    )

    _assert_shapes(terms, nstates=args.states, walkers=args.walkers)
    print("ferminet_pbc_penalty_terms: ok")
    print(f"experiment: {bundle.paths.experiment.relative_to(PROJECT_ROOT)}")
    print(f"jax_platform: {jax.default_backend()}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"external_state_params: {len(state_params)}")
    print(f"local_energy_source: {'real_pbc' if args.evaluate_local_energy else 'cheap'}")
    print(f"state_energy_shape: {tuple(terms['state_energy'].shape)}")
    print(f"overlap_matrix_shape: {tuple(terms['overlap_matrix'].shape)}")
    print(f"psi_ratio_shape: {tuple(terms['psi_ratio'].shape)}")
    print(f"local_energy_shape: {tuple(terms['local_energy'].shape)}")
    print(f"penalty_objective_shape: {tuple(terms['penalty_objective'].shape)}")
    print(f"collapse_flag_shape: {tuple(terms['collapse_flag'].shape)}")
    return 0


def _cheap_local_energy(adapter):
    jnp = adapter.modules.jnp

    def local_energy(params, positions, spins, atoms, charges):
        del params, atoms, charges
        position_scale = jnp.mean(positions * positions, axis=-1)
        spin_scale = 0.01 * jnp.sum(spins, axis=-1)
        return position_scale + spin_scale

    return local_energy


def _assert_shapes(terms, *, nstates: int, walkers: int) -> None:
    expected = {
        "state_energy": (nstates,),
        "overlap_matrix": (nstates, nstates),
        "psi_ratio": (nstates, nstates, walkers),
        "local_energy": (nstates, walkers),
        "penalty_objective": (),
        "collapse_flag": (),
    }
    for key, shape in expected.items():
        if tuple(terms[key].shape) != shape:
            raise ValueError(f"{key} shape {tuple(terms[key].shape)} != {shape}")


if __name__ == "__main__":
    raise SystemExit(main())
