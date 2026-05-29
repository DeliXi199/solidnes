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
from solidnes.excited_states.penalty import overlap_gradient_scale
from solidnes.excited_states.penalty import penalty_vmc_terms
from solidnes.backends.ferminet_psiformer_attention import (
    install_psiformer_attention_implementation,
)
from solidnes.backends.ferminet_psiformer_attention import (
    psiformer_kwargs_from_config,
)


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
        install_psiformer_attention_implementation(cfg)
        return psiformer.make_fermi_net(
            cfg.system.electrons,
            charges,
            **common_kwargs,
            **psiformer_kwargs_from_config(cfg.network.psiformer),
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
    ratio_clip_width: float | None = None,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = None,
    overlap_scale_by: str | None = None,
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    overlap_scale_state_energy: Any | None = None,
    overlap_scale_state_energy_std: Any | None = None,
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
        ratio_clip_width=ratio_clip_width,
        ratio_exclude_width=ratio_exclude_width,
        max_logabs_ratio=max_logabs_ratio,
    )
    energy_std = adapter.modules.jnp.std(energy.local_energy, axis=-1)
    scale_energy = (
        energy.state_energy
        if overlap_scale_state_energy is None
        else adapter.modules.jnp.asarray(overlap_scale_state_energy)
    )
    scale_std = (
        energy_std
        if overlap_scale_state_energy_std is None
        else adapter.modules.jnp.asarray(overlap_scale_state_energy_std)
    )
    scale = overlap_gradient_scale(
        scale_energy,
        scale_std,
        scale_by=overlap_scale_by,
        min_gap_scale_factor=min_gap_scale_factor,
        max_scale_factor=max_scale_factor,
    )
    terms = penalty_vmc_terms(
        energy.state_energy,
        overlap["overlap_matrix"],
        penalty_alpha=penalty_alpha,
        energy_weights=energy_weights,
        overlap_scale=scale,
    )
    return {
        "local_energy": energy.local_energy,
        "state_energy": energy.state_energy,
        "state_energy_std": energy_std,
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
    ratio_clip_width: float | None = None,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = None,
    overlap_scale_by: str | None = None,
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    overlap_scale_state_energy: Any | None = None,
    overlap_scale_state_energy_std: Any | None = None,
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
        ratio_clip_width=ratio_clip_width,
        ratio_exclude_width=ratio_exclude_width,
        max_logabs_ratio=max_logabs_ratio,
        overlap_scale_by=overlap_scale_by,
        min_gap_scale_factor=min_gap_scale_factor,
        max_scale_factor=max_scale_factor,
        overlap_scale_state_energy=overlap_scale_state_energy,
        overlap_scale_state_energy_std=overlap_scale_state_energy_std,
    )["penalty_objective"]


def ferminet_pbc_penalty_training_objective(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
    ratio_clip_width: float | None = 10.0,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = 30.0,
    local_energy_clip_width: float | None = 5.0,
    local_energy_exclude_width: float = float("inf"),
    overlap_scale_by: str | None = "max_gap_std",
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    state_ordering: str = "index",
    precomputed_terms: dict[str, Any] | None = None,
    overlap_scale_state_energy: Any | None = None,
    overlap_scale_state_energy_std: Any | None = None,
) -> Any:
    """Return a penalty objective with a paper-style optimization tangent.

    The returned scalar has the diagnostic value of the penalty objective, while
    its gradient avoids differentiating through the local-energy operator.  The
    energy tangent uses a VMC score-function surrogate, and overlap gradients are
    ordered so lower-index states are treated as fixed references.
    """

    if precomputed_terms is None:
        terms = evaluate_ferminet_pbc_penalty_terms(
            adapter,
            state_params,
            samples,
            penalty_alpha=penalty_alpha,
            local_energy=local_energy,
            energy_weights=energy_weights,
            collapse_threshold=collapse_threshold,
            clip_upper=clip_upper,
            ratio_clip_width=ratio_clip_width,
            ratio_exclude_width=ratio_exclude_width,
            max_logabs_ratio=max_logabs_ratio,
            overlap_scale_by=overlap_scale_by,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
            overlap_scale_state_energy=overlap_scale_state_energy,
            overlap_scale_state_energy_std=overlap_scale_state_energy_std,
        )
    else:
        terms = precomputed_terms
    true_objective = terms["penalty_objective"]
    energy_surrogate = _state_energy_score_surrogate(
        adapter,
        state_params,
        samples,
        adapter.modules.jax.lax.stop_gradient(terms["local_energy"]),
        energy_weights=energy_weights,
        local_energy_clip_width=local_energy_clip_width,
        local_energy_exclude_width=local_energy_exclude_width,
    )
    overlap_surrogate = _ordered_overlap_surrogate(
        adapter,
        state_params,
        samples,
        adapter.modules.jax.lax.stop_gradient(terms["overlap_gradient_scale"]),
        clip_upper=clip_upper,
        ratio_clip_width=ratio_clip_width,
        ratio_exclude_width=ratio_exclude_width,
        max_logabs_ratio=max_logabs_ratio,
        state_ordering=state_ordering,
    )
    surrogate = energy_surrogate + penalty_alpha * overlap_surrogate
    jax = adapter.modules.jax
    return jax.lax.stop_gradient(true_objective) + _zero_primal_tangent(
        adapter,
        surrogate,
    )


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
    ratio_clip_width: float | None = 10.0,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = 30.0,
    local_energy_clip_width: float | None = 5.0,
    local_energy_exclude_width: float = float("inf"),
    overlap_scale_by: str | None = "max_gap_std",
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    state_ordering: str = "index",
    gradient_mode: str = "paper_tangent",
    precomputed_terms: dict[str, Any] | None = None,
    overlap_scale_state_energy: Any | None = None,
    overlap_scale_state_energy_std: Any | None = None,
) -> tuple[Any, Any]:
    """Evaluate the scalar penalty objective and gradients over state params."""

    def objective(params):
        if gradient_mode == "direct":
            if precomputed_terms is not None:
                raise ValueError("precomputed_terms are only supported for paper_tangent")
            return ferminet_pbc_penalty_objective(
                adapter,
                params,
                samples,
                penalty_alpha=penalty_alpha,
                local_energy=local_energy,
                energy_weights=energy_weights,
                collapse_threshold=collapse_threshold,
                clip_upper=clip_upper,
                ratio_clip_width=ratio_clip_width,
                ratio_exclude_width=ratio_exclude_width,
                max_logabs_ratio=max_logabs_ratio,
                overlap_scale_by=overlap_scale_by,
                min_gap_scale_factor=min_gap_scale_factor,
                max_scale_factor=max_scale_factor,
                overlap_scale_state_energy=overlap_scale_state_energy,
                overlap_scale_state_energy_std=overlap_scale_state_energy_std,
            )
        if gradient_mode != "paper_tangent":
            raise ValueError(f"Unsupported gradient_mode: {gradient_mode}")
        return ferminet_pbc_penalty_training_objective(
            adapter,
            params,
            samples,
            penalty_alpha=penalty_alpha,
            local_energy=local_energy,
            energy_weights=energy_weights,
            collapse_threshold=collapse_threshold,
            clip_upper=clip_upper,
            ratio_clip_width=ratio_clip_width,
            ratio_exclude_width=ratio_exclude_width,
            max_logabs_ratio=max_logabs_ratio,
            local_energy_clip_width=local_energy_clip_width,
            local_energy_exclude_width=local_energy_exclude_width,
            overlap_scale_by=overlap_scale_by,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
            state_ordering=state_ordering,
            precomputed_terms=precomputed_terms,
            overlap_scale_state_energy=overlap_scale_state_energy,
            overlap_scale_state_energy_std=overlap_scale_state_energy_std,
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


def _state_energy_score_surrogate(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    local_energy_values: Any,
    *,
    energy_weights: Any | None,
    local_energy_clip_width: float | None,
    local_energy_exclude_width: float,
) -> Any:
    """Build a VMC score-function surrogate for state-energy gradients."""

    jnp = adapter.modules.jnp
    jax = adapter.modules.jax
    contributions = []
    for state_idx, params in enumerate(state_params):
        positions, spins, atoms, charges = samples.for_sample_state(state_idx)
        _, logabs = adapter.batched_signed_network(
            params,
            positions,
            spins,
            atoms,
            charges,
        )
        local_energy, gradient_mask = _clip_local_energy_by_median(
            adapter,
            local_energy_values[state_idx],
            clip_width=local_energy_clip_width,
            exclude_width=local_energy_exclude_width,
        )
        mean_energy = _masked_mean(jnp, local_energy, gradient_mask)
        centered_energy = jax.lax.stop_gradient(local_energy - mean_energy)
        centered_logabs = logabs - jax.lax.stop_gradient(jnp.mean(logabs))
        contributions.append(
            _masked_mean(jnp, centered_energy * centered_logabs, gradient_mask)
        )
    per_state = jnp.stack(contributions, axis=0)
    weights = _normalized_state_weights(adapter, energy_weights, len(state_params))
    return jnp.sum(per_state * weights)


def _ordered_overlap_surrogate(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    overlap_scale: Any,
    *,
    clip_upper: bool,
    ratio_clip_width: float | None,
    ratio_exclude_width: float,
    max_logabs_ratio: float | None,
    state_ordering: str,
) -> Any:
    """Return ordered overlap loss with lower-state stop-gradient behavior."""

    if state_ordering not in ("index", "natural"):
        raise ValueError("Only index/natural state ordering is supported for now")
    jnp = adapter.modules.jnp
    total = jnp.asarray(0.0)
    nstates = len(state_params)
    stopped_params = tuple(_stop_gradient_tree(adapter, params) for params in state_params)
    for low_idx in range(nstates):
        for high_idx in range(low_idx + 1, nstates):
            low_params = stopped_params[low_idx]
            high_params = state_params[high_idx]
            ratio_low_high = _pair_wavefunction_ratio(
                adapter,
                low_params,
                high_params,
                samples,
                sample_state=high_idx,
                max_logabs_ratio=max_logabs_ratio,
            )
            ratio_high_low = _pair_wavefunction_ratio(
                adapter,
                high_params,
                low_params,
                samples,
                sample_state=low_idx,
                max_logabs_ratio=max_logabs_ratio,
            )
            ratio_low_high = _clip_pair_ratio(
                adapter,
                ratio_low_high,
                ratio_clip_width=ratio_clip_width,
                ratio_exclude_width=ratio_exclude_width,
            )
            ratio_high_low = _clip_pair_ratio(
                adapter,
                ratio_high_low,
                ratio_clip_width=ratio_clip_width,
                ratio_exclude_width=ratio_exclude_width,
            )
            product = jnp.mean(ratio_low_high) * jnp.mean(ratio_high_low)
            if clip_upper:
                pair_penalty = jnp.clip(product, 0.0, 1.0)
            else:
                pair_penalty = jnp.maximum(product, 0.0)
            scale = _pair_scale(adapter, overlap_scale, low_idx, high_idx)
            total = total + scale * pair_penalty
    return total


def _pair_wavefunction_ratio(
    adapter: FermiNetPBCExternalStateAdapter,
    numerator_params: Any,
    denominator_params: Any,
    samples: FermiNetPBCStateSamples,
    *,
    sample_state: int,
    max_logabs_ratio: float | None,
) -> Any:
    """Evaluate psi_num(r_sample_state) / psi_den(r_sample_state)."""

    jnp = adapter.modules.jnp
    positions, spins, atoms, charges = samples.for_sample_state(sample_state)
    numerator_sign, numerator_logabs = adapter.batched_signed_network(
        numerator_params,
        positions,
        spins,
        atoms,
        charges,
    )
    denominator_sign, denominator_logabs = adapter.batched_signed_network(
        denominator_params,
        positions,
        spins,
        atoms,
        charges,
    )
    log_ratio = numerator_logabs - denominator_logabs
    if max_logabs_ratio is not None:
        log_ratio = jnp.clip(log_ratio, -max_logabs_ratio, max_logabs_ratio)
    ratio = (numerator_sign / denominator_sign) * jnp.exp(log_ratio)
    return jnp.real(ratio)


def _clip_pair_ratio(
    adapter: FermiNetPBCExternalStateAdapter,
    ratio: Any,
    *,
    ratio_clip_width: float | None,
    ratio_exclude_width: float,
) -> Any:
    if ratio_clip_width is None:
        return ratio
    jnp = adapter.modules.jnp
    jax = adapter.modules.jax
    center = jax.lax.stop_gradient(jnp.median(ratio, axis=-1, keepdims=True))
    deviation = jnp.abs(ratio - center)
    sigma = jax.lax.stop_gradient(jnp.median(deviation, axis=-1, keepdims=True))
    clipped = jnp.clip(
        ratio,
        center - ratio_clip_width * sigma,
        center + ratio_clip_width * sigma,
    )
    mask = deviation < ratio_exclude_width
    return jnp.where(mask, clipped, jax.lax.stop_gradient(clipped))


def _clip_local_energy_by_median(
    adapter: FermiNetPBCExternalStateAdapter,
    local_energy: Any,
    *,
    clip_width: float | None,
    exclude_width: float,
) -> tuple[Any, Any]:
    jnp = adapter.modules.jnp
    values = jnp.real(jnp.asarray(local_energy))
    finite = jnp.isfinite(values)
    safe = jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_width is None:
        return safe, finite
    center = jnp.median(safe, axis=-1, keepdims=True)
    deviation = jnp.abs(safe - center)
    mad = jnp.mean(deviation, axis=-1, keepdims=True)
    clipped = jnp.clip(safe, center - clip_width * mad, center + clip_width * mad)
    gradient_mask = finite & (deviation < exclude_width)
    return clipped, gradient_mask


def _zero_primal_tangent(adapter: FermiNetPBCExternalStateAdapter, value: Any) -> Any:
    """Return a zero-valued tangent term with a finite primal value."""

    jnp = adapter.modules.jnp
    jax = adapter.modules.jax
    finite_value = jnp.nan_to_num(
        jnp.real(jnp.asarray(value)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return finite_value - jax.lax.stop_gradient(finite_value)


def _masked_mean(jnp: Any, value: Any, mask: Any) -> Any:
    weighted = jnp.where(mask, value, 0.0)
    denom = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sum(weighted) / denom


def _normalized_state_weights(
    adapter: FermiNetPBCExternalStateAdapter,
    energy_weights: Any | None,
    nstates: int,
) -> Any:
    jnp = adapter.modules.jnp
    if energy_weights is None:
        return jnp.ones((nstates,)) / nstates
    weights = jnp.asarray(energy_weights)
    if weights.shape != (nstates,):
        raise ValueError("energy_weights must have shape [states]")
    return weights / jnp.sum(weights)


def _pair_scale(
    adapter: FermiNetPBCExternalStateAdapter,
    overlap_scale: Any,
    low_idx: int,
    high_idx: int,
) -> Any:
    jnp = adapter.modules.jnp
    scale = jnp.asarray(overlap_scale)
    if scale.ndim == 0:
        value = scale
    else:
        value = scale[low_idx, high_idx]
    return adapter.modules.jax.lax.stop_gradient(value)


def _stop_gradient_tree(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    return adapter.modules.jax.tree_util.tree_map(
        adapter.modules.jax.lax.stop_gradient,
        tree,
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
    "ferminet_pbc_penalty_training_objective",
    "init_external_state_params",
    "load_ferminet_jax_modules",
    "make_network_from_config",
    "make_tiny_state_samples",
    "value_and_grad_ferminet_pbc_penalty_objective",
    "wrap_pbc_local_energy",
    "wrap_signed_network",
]
