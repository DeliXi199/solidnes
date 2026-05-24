#!/usr/bin/env python
"""Build-only FermiNet PBC adapter check for external excited-state params.

This script verifies the first real FermiNet/JAX link for the SolidNES
penalty-state scaffold without using FermiNet's molecular excited-state path.
It intentionally keeps ``cfg.system.states == 0`` because the upstream PBC
Hamiltonian rejects ``states > 0``.
"""

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
    parser.add_argument(
        "--states",
        type=int,
        default=2,
        help="Number of externally managed state parameter trees.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=1,
        help="Number of tiny synthetic walkers per state for network shape check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="JAX PRNG seed.",
    )
    parser.add_argument(
        "--platform",
        default="cpu",
        help="JAX platform for this build-only check. Use '' to leave unset.",
    )
    parser.add_argument(
        "--evaluate-local-energy",
        action="store_true",
        help="Also evaluate the wrapped local energy. This is more expensive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.states < 2:
        raise ValueError("--states must be at least 2 for this adapter check")
    if args.walkers < 1:
        raise ValueError("--walkers must be at least 1")
    if args.platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)
        os.environ.setdefault("JAX_PLATFORMS", args.platform)

    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        assert_pbc_external_state_config,
        build_external_state_adapter,
        configure_jax_platform,
    )
    from solidnes.excited_states.ferminet_pbc_scaffold import (
        evaluate_state_energy_estimate,
        evaluate_state_wavefunction_matrix,
    )

    configure_jax_platform(args.platform)
    bundle = build_ferminet_adapter(args.experiment)
    cfg = bundle.cfg
    cfg_states = assert_pbc_external_state_config(cfg)
    adapter = build_external_state_adapter(cfg, platform=args.platform)

    jax = adapter.modules.jax
    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key = jax.random.split(key, 3)
    state_params = adapter.init_state_params(params_key, args.states)
    state_samples = adapter.tiny_state_samples(
        samples_key,
        nstates=args.states,
        walkers=args.walkers,
    )

    matrix = evaluate_state_wavefunction_matrix(
        adapter.batched_signed_network,
        state_params,
        state_samples,
    )

    print("ferminet_pbc_excited_adapter_build: ok")
    print(f"experiment: {bundle.paths.experiment.relative_to(PROJECT_ROOT)}")
    print(f"jax_platform: {jax.default_backend()}")
    print(f"network_type: {cfg.network.network_type}")
    print(f"external_state_params: {len(state_params)}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"nelec: {sum(adapter.nspins)}")
    print(f"natoms: {adapter.atoms.shape[0]}")
    print(f"walkers_per_state: {args.walkers}")
    print(f"wavefunction_sign_shape: {tuple(matrix.sign.shape)}")
    print(f"wavefunction_logabs_shape: {tuple(matrix.logabs.shape)}")
    print("local_energy_wrapped: yes")

    if args.evaluate_local_energy:
        energy = evaluate_state_energy_estimate(
            adapter.batched_local_energy,
            state_params,
            state_samples,
        )
        print(f"local_energy_shape: {tuple(energy.local_energy.shape)}")
        print(f"state_energy_shape: {tuple(energy.state_energy.shape)}")
    else:
        print("local_energy_evaluated: no")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
