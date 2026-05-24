"""FermiNet/JAX adapter helpers for PBC excited-state scaffolds.

The upstream FermiNet PBC Hamiltonian only supports the ground-state network
interface.  SolidNES therefore keeps ``cfg.system.states == 0`` and manages one
parameter tree per target state outside FermiNet.

This module intentionally imports FermiNet and JAX lazily so the rest of
``solidnes.excited_states`` remains importable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any

from solidnes.excited_states.ferminet_pbc_scaffold import BatchedLocalEnergy
from solidnes.excited_states.ferminet_pbc_scaffold import BatchedSignedNetwork
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples
from solidnes.excited_states.ferminet_pbc_scaffold import broadcast_state_samples
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_overlap_diagnostics
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_energy_estimate
from solidnes.excited_states.penalty import penalty_vmc_terms


@dataclass(frozen=True)
class FermiNetJAXModules:
    """JAX and FermiNet modules loaded for a real adapter build."""

    jax: Any
    jnp: Any
    networks: Any


@dataclass(frozen=True)
class FermiNetPBCExternalStateAdapter:
    """Real FermiNet PBC callables wrapped for external state management."""

    cfg: Any
    modules: FermiNetJAXModules
    atoms: Any
    charges: Any
    nspins: tuple[int, int]
    network: Any
    signed_network: Any
    batched_signed_network: BatchedSignedNetwork
    batched_local_energy: BatchedLocalEnergy

    def init_state_params(self, key: Any, nstates: int) -> tuple[Any, ...]:
        """Initialize one external FermiNet parameter tree per state."""

        return init_external_state_params(
            self.network,
            self.modules.jax,
            key,
            nstates,
        )

    def tiny_state_samples(
        self,
        key: Any,
        *,
        nstates: int,
        walkers: int,
    ) -> FermiNetPBCStateSamples:
        """Build tiny state-indexed samples for adapter shape checks."""

        return make_tiny_state_samples(
            jax=self.modules.jax,
            jnp=self.modules.jnp,
            key=key,
            nstates=nstates,
            walkers=walkers,
            nspins=self.nspins,
            atoms=self.atoms,
            charges=self.charges,
            lattice=self.modules.jnp.asarray(
                self.cfg.system.make_local_energy_kwargs["lattice"]
            ),
        )


def configure_jax_platform(platform: str | None = "cpu") -> None:
    """Set the JAX platform before importing JAX."""

    if platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)


def load_ferminet_jax_modules(platform: str | None = "cpu") -> FermiNetJAXModules:
    """Load JAX/FermiNet and apply SolidNES compatibility shims."""

    configure_jax_platform(platform)

    from solidnes.backends.ferminet_jax_compat import apply_modern_jax_shims

    apply_modern_jax_shims()

    import jax
    import jax.numpy as jnp
    from ferminet import networks

    return FermiNetJAXModules(jax=jax, jnp=jnp, networks=networks)


def assert_pbc_external_state_config(cfg: Any) -> int:
    """Validate that FermiNet's internal excited-state path is disabled."""

    cfg_states = int(cfg.system.get("states", 0) or 0)
    if cfg_states != 0:
        raise ValueError(
            "FermiNet PBC excited-state adapter must keep cfg.system.states == 0; "
            f"got {cfg_states}"
        )
    return cfg_states


def build_external_state_adapter(
    cfg: Any,
    *,
    platform: str | None = "cpu",
) -> FermiNetPBCExternalStateAdapter:
    """Build real FermiNet PBC callables for externally managed states."""

    assert_pbc_external_state_config(cfg)
    modules = load_ferminet_jax_modules(platform)
    jnp = modules.jnp

    atoms = jnp.asarray([atom.coords for atom in cfg.system.molecule])
    charges = jnp.asarray([atom.charge for atom in cfg.system.molecule])
    nspins = tuple(int(x) for x in cfg.system.electrons)
    network = make_network_from_config(cfg, charges)
    signed_network = network.apply
    return FermiNetPBCExternalStateAdapter(
        cfg=cfg,
        modules=modules,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        network=network,
        signed_network=signed_network,
        batched_signed_network=wrap_signed_network(modules.jax, signed_network),
        batched_local_energy=wrap_pbc_local_energy(
            cfg=cfg,
            jax=modules.jax,
            networks=modules.networks,
            signed_network=signed_network,
            charges=charges,
            nspins=nspins,
        ),
    )


def make_network_from_config(cfg: Any, charges: Any) -> Any:
    """Construct a real FermiNet network from an already-built PBC config."""

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


def evaluate_ferminet_pbc_penalty_terms(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
) -> dict[str, Any]:
    """Evaluate real FermiNet PBC diagnostics and penalty objective terms.

    By default this uses ``adapter.batched_local_energy``.  Build-only checks can
    pass a cheap local-energy stand-in to validate the full penalty data path
    without triggering the expensive PBC Laplacian.
    """

    active_local_energy = local_energy or adapter.batched_local_energy
    energy = evaluate_state_energy_estimate(active_local_energy, state_params, samples)
    overlap = evaluate_overlap_diagnostics(
        adapter.batched_signed_network,
        state_params,
        samples,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )
    terms = penalty_vmc_terms(
        energy.state_energy,
        overlap["overlap_matrix"],
        penalty_alpha=penalty_alpha,
        energy_weights=energy_weights,
    )
    return {
        "local_energy": energy.local_energy,
        "state_energy": energy.state_energy,
        **overlap,
        **terms,
    }


def ferminet_pbc_penalty_objective(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
) -> Any:
    """Return the scalar FermiNet PBC external-state penalty objective."""

    return evaluate_ferminet_pbc_penalty_terms(
        adapter,
        state_params,
        samples,
        penalty_alpha=penalty_alpha,
        local_energy=local_energy,
        energy_weights=energy_weights,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )["penalty_objective"]


def value_and_grad_ferminet_pbc_penalty_objective(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
) -> tuple[Any, Any]:
    """Evaluate the scalar penalty objective and gradients over state params."""

    def objective(params):
        return ferminet_pbc_penalty_objective(
            adapter,
            params,
            samples,
            penalty_alpha=penalty_alpha,
            local_energy=local_energy,
            energy_weights=energy_weights,
            collapse_threshold=collapse_threshold,
            clip_upper=clip_upper,
        )

    return adapter.modules.jax.value_and_grad(objective)(state_params)


def apply_external_state_sgd_step(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    grads: Any,
    *,
    learning_rate: float,
) -> tuple[Any, ...]:
    """Apply one simple SGD update to externally managed state params."""

    jax = adapter.modules.jax
    lr = adapter.modules.jnp.asarray(learning_rate)
    return jax.tree_util.tree_map(
        lambda param, grad: param - lr * grad,
        state_params,
        grads,
    )


def init_external_state_params(
    network: Any,
    jax: Any,
    key: Any,
    nstates: int,
) -> tuple[Any, ...]:
    """Initialize one externally managed FermiNet parameter tree per state."""

    if nstates < 2:
        raise ValueError("nstates must be at least 2")
    param_keys = jax.random.split(key, nstates)
    return tuple(network.init(subkey) for subkey in param_keys)


def make_tiny_state_samples(
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
) -> FermiNetPBCStateSamples:
    """Create tiny PBC samples with leading axes ``[states, walkers]``."""

    if nstates < 2:
        raise ValueError("nstates must be at least 2")
    if walkers < 1:
        raise ValueError("walkers must be at least 1")
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
    return broadcast_state_samples(
        positions=positions,
        spins=jnp.broadcast_to(one_spin, (nstates, walkers, nelec)),
        atoms=jnp.broadcast_to(atoms, (nstates, walkers) + atoms.shape),
        charges=jnp.broadcast_to(charges, (nstates, walkers) + charges.shape),
    )


def wrap_signed_network(jax: Any, signed_network: Any) -> BatchedSignedNetwork:
    """Wrap FermiNet's single-walker ``network.apply`` in a walker vmap."""

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


def wrap_pbc_local_energy(
    *,
    cfg: Any,
    jax: Any,
    networks: Any,
    signed_network: Any,
    charges: Any,
    nspins: tuple[int, int],
    key_seed: int = 715,
) -> BatchedLocalEnergy:
    """Wrap FermiNet's PBC local energy for state-owned walker batches."""

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
        keys = jax.random.split(jax.random.PRNGKey(key_seed), positions.shape[0])

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


__all__ = [
    "FermiNetJAXModules",
    "FermiNetPBCExternalStateAdapter",
    "apply_external_state_sgd_step",
    "assert_pbc_external_state_config",
    "build_external_state_adapter",
    "configure_jax_platform",
    "evaluate_ferminet_pbc_penalty_terms",
    "ferminet_pbc_penalty_objective",
    "init_external_state_params",
    "load_ferminet_jax_modules",
    "make_network_from_config",
    "make_tiny_state_samples",
    "value_and_grad_ferminet_pbc_penalty_objective",
    "wrap_pbc_local_energy",
    "wrap_signed_network",
]
