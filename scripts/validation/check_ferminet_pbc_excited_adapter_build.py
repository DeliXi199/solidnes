#!/usr/bin/env python
"""Build-only FermiNet PBC adapter check for external excited-state params.

This script verifies the first real FermiNet/JAX link for the SolidNES
penalty-state scaffold without using FermiNet's molecular excited-state path.
It intentionally keeps ``cfg.system.states == 0`` because the upstream PBC
Hamiltonian rejects ``states > 0``.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any


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
    from solidnes.backends.ferminet_jax_compat import apply_modern_jax_shims
    from solidnes.excited_states.ferminet_pbc_scaffold import (
        broadcast_state_samples,
        evaluate_state_energy_estimate,
        evaluate_state_wavefunction_matrix,
    )

    apply_modern_jax_shims()

    import jax
    import jax.numpy as jnp
    from ferminet import networks

    bundle = build_ferminet_adapter(args.experiment)
    cfg = bundle.cfg
    cfg_states = int(cfg.system.get("states", 0) or 0)
    if cfg_states != 0:
        raise ValueError(
            "FermiNet PBC excited-state adapter must keep cfg.system.states == 0; "
            f"got {cfg_states}"
        )

    atoms = jnp.asarray([atom.coords for atom in cfg.system.molecule])
    charges = jnp.asarray([atom.charge for atom in cfg.system.molecule])
    nspins = tuple(int(x) for x in cfg.system.electrons)
    network = _make_network(cfg, charges)
    signed_network = network.apply

    key = jax.random.PRNGKey(args.seed)
    _, params_key, samples_key = jax.random.split(key, 3)
    param_keys = jax.random.split(params_key, args.states)
    state_params = tuple(network.init(subkey) for subkey in param_keys)

    samples = _make_tiny_samples(
        jax=jax,
        jnp=jnp,
        key=samples_key,
        nstates=args.states,
        walkers=args.walkers,
        nspins=nspins,
        atoms=atoms,
        charges=charges,
        lattice=jnp.asarray(cfg.system.make_local_energy_kwargs["lattice"]),
    )
    state_samples = broadcast_state_samples(
        positions=samples["positions"],
        spins=samples["spins"],
        atoms=samples["atoms"],
        charges=samples["charges"],
    )

    batched_signed_network = _wrap_signed_network(jax, signed_network)
    matrix = evaluate_state_wavefunction_matrix(
        batched_signed_network,
        state_params,
        state_samples,
    )
    local_energy = _wrap_local_energy(
        cfg=cfg,
        jax=jax,
        networks=networks,
        signed_network=signed_network,
        charges=charges,
        nspins=nspins,
    )

    print("ferminet_pbc_excited_adapter_build: ok")
    print(f"experiment: {bundle.paths.experiment.relative_to(PROJECT_ROOT)}")
    print(f"jax_platform: {jax.default_backend()}")
    print(f"network_type: {cfg.network.network_type}")
    print(f"external_state_params: {len(state_params)}")
    print(f"cfg_system_states: {cfg_states}")
    print(f"nelec: {sum(nspins)}")
    print(f"natoms: {atoms.shape[0]}")
    print(f"walkers_per_state: {args.walkers}")
    print(f"wavefunction_sign_shape: {tuple(matrix.sign.shape)}")
    print(f"wavefunction_logabs_shape: {tuple(matrix.logabs.shape)}")
    print("local_energy_wrapped: yes")

    if args.evaluate_local_energy:
        energy = evaluate_state_energy_estimate(
            local_energy,
            state_params,
            state_samples,
        )
        print(f"local_energy_shape: {tuple(energy.local_energy.shape)}")
        print(f"state_energy_shape: {tuple(energy.state_energy.shape)}")
    else:
        print("local_energy_evaluated: no")

    return 0


def _make_network(cfg: Any, charges: Any) -> Any:
    from ferminet import envelopes
    from ferminet import networks
    from ferminet import psiformer

    if cfg.network.get("make_feature_layer_fn"):
        module_name, fn_name = cfg.network.make_feature_layer_fn.rsplit(".", 1)
        make_feature_layer = getattr(importlib.import_module(module_name), fn_name)
        feature_layer = make_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs,
        )
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get("rescale_inputs", False),
        )

    if cfg.network.get("make_envelope_fn"):
        module_name, fn_name = cfg.network.make_envelope_fn.rsplit(".", 1)
        make_envelope = getattr(importlib.import_module(module_name), fn_name)
        envelope = make_envelope(**cfg.network.make_envelope_kwargs)
    else:
        envelope = envelopes.make_isotropic_envelope()

    use_complex = bool(cfg.network.get("complex", False))
    common_kwargs = dict(
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=0,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get("jastrow", "default"),
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.get("rescale_inputs", False),
        complex_output=use_complex,
    )
    if cfg.network.network_type == "ferminet":
        return networks.make_fermi_net(
            cfg.system.electrons,
            charges,
            full_det=cfg.network.full_det,
            **common_kwargs,
            **cfg.network.ferminet,
        )
    if cfg.network.network_type == "psiformer":
        return psiformer.make_fermi_net(
            cfg.system.electrons,
            charges,
            **common_kwargs,
            **cfg.network.psiformer,
        )
    raise ValueError(f"Unsupported FermiNet network_type: {cfg.network.network_type}")


def _make_tiny_samples(
    *,
    jax: Any,
    jnp: Any,
    key: Any,
    nstates: int,
    walkers: int,
    nspins: tuple[int, int],
    atoms: Any,
    charges: Any,
    lattice: Any,
) -> dict[str, Any]:
    nelec = sum(nspins)
    ndim = int(lattice.shape[0])
    frac = jax.random.uniform(
        key,
        (nstates, walkers, nelec, ndim),
        minval=0.05,
        maxval=0.95,
    )
    positions = jnp.einsum("...i,ji->...j", frac, lattice).reshape(
        nstates,
        walkers,
        nelec * ndim,
    )
    one_spin = jnp.concatenate((jnp.ones(nspins[0]), -jnp.ones(nspins[1])))
    spins = jnp.broadcast_to(one_spin, (nstates, walkers, nelec))
    atom_batch = jnp.broadcast_to(atoms, (nstates, walkers) + atoms.shape)
    charge_batch = jnp.broadcast_to(charges, (nstates, walkers) + charges.shape)
    return {
        "positions": positions,
        "spins": spins,
        "atoms": atom_batch,
        "charges": charge_batch,
    }


def _wrap_signed_network(jax: Any, signed_network: Any) -> Any:
    def batched_signed_network(params, positions, spins, atoms, charges):
        return jax.vmap(
            lambda pos, spin, atom, charge: signed_network(
                params,
                pos,
                spin,
                atom,
                charge,
            ),
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(positions, spins, atoms, charges)

    return batched_signed_network


def _wrap_local_energy(
    *,
    cfg: Any,
    jax: Any,
    networks: Any,
    signed_network: Any,
    charges: Any,
    nspins: tuple[int, int],
) -> Any:
    if cfg.system.get("make_local_energy_fn"):
        module_name, fn_name = cfg.system.make_local_energy_fn.rsplit(".", 1)
        make_local_energy = getattr(importlib.import_module(module_name), fn_name)
    else:
        from ferminet import hamiltonian

        make_local_energy = hamiltonian.local_energy

    pp_symbols = cfg.system.get("pp", {"symbols": None}).get("symbols")
    local_energy_fn = make_local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        ndim=cfg.system.ndim,
        complex_output=bool(cfg.network.get("complex", False)),
        laplacian_method=cfg.optim.get("laplacian", "default"),
        states=0,
        state_specific=False,
        pp_type=cfg.system.get("pp", {"type": "ccecp"}).get("type"),
        pp_symbols=pp_symbols if cfg.system.get("use_pp") else None,
        **cfg.system.make_local_energy_kwargs,
    )

    def batched_local_energy(params, positions, spins, atoms, charges):
        keys = jax.random.split(jax.random.PRNGKey(715), positions.shape[0])

        def one_walker(key, pos, spin, atom, charge):
            data = networks.FermiNetData(
                positions=pos,
                spins=spin,
                atoms=atom,
                charges=charge,
            )
            value, _ = local_energy_fn(params, key, data)
            return value

        return jax.vmap(one_walker, in_axes=(0, 0, 0, 0, 0))(
            keys,
            positions,
            spins,
            atoms,
            charges,
        )

    return batched_local_energy


if __name__ == "__main__":
    raise SystemExit(main())
