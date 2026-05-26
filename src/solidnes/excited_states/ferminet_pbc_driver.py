"""Production-oriented driver helpers for FermiNet PBC excited states."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from solidnes.excited_states.ferminet_pbc_adapter import (
    FermiNetPBCExternalStateAdapter,
)
from solidnes.excited_states.ferminet_pbc_adapter import (
    evaluate_ferminet_pbc_penalty_terms,
)
from solidnes.excited_states.ferminet_pbc_scaffold import BatchedLocalEnergy
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples
from solidnes.excited_states.ferminet_pbc_scaffold import broadcast_state_samples
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStatePenaltyStepDiagnostics,
)
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStateOptimizerState,
)
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStatePenaltyRunningStats,
)
from solidnes.excited_states.ferminet_pbc_training import (
    external_state_penalty_optimizer_step,
)
from solidnes.excited_states.ferminet_pbc_training import block_tree_until_ready
from solidnes.excited_states.ferminet_pbc_training import (
    merge_external_optimizer_state,
    merge_external_state_params,
)
from solidnes.excited_states.ferminet_pbc_training import (
    update_external_state_penalty_running_stats,
)


@dataclass(frozen=True)
class ExternalStateMetropolisDiagnostics:
    """Diagnostics for one external-state sampling block."""

    iteration: int
    steps: int
    acceptance: Any


@dataclass(frozen=True)
class ExternalStateDriverStepDiagnostics:
    """Diagnostics for one sampler-plus-optimizer driver iteration."""

    iteration: int
    sampler: ExternalStateMetropolisDiagnostics
    update: ExternalStatePenaltyStepDiagnostics


@dataclass(frozen=True)
class ExternalStateDriverResult:
    """Result of a multi-iteration external-state penalty-VMC driver run."""

    state_params: tuple[Any, ...]
    samples: FermiNetPBCStateSamples
    sampler_key: Any
    initial_terms: dict[str, Any]
    final_terms: dict[str, Any]
    history: tuple[ExternalStateDriverStepDiagnostics, ...]
    optimizer_state: ExternalStateOptimizerState | None = None
    running_stats: ExternalStatePenaltyRunningStats | None = None


def run_external_state_penalty_driver(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    sampler_key: Any,
    iterations: int,
    learning_rate: float,
    penalty_alpha: float,
    sampler_burn_in: int = 0,
    sampler_steps_per_iteration: int = 1,
    sampler_proposal_width: float = 0.02,
    sampler_wrap: bool = True,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    optimizer_name: str = "sgd",
    optimizer_state: ExternalStateOptimizerState | None = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    lamb_eps: float = 1.0e-6,
    kfac_damping: float | None = None,
    kfac_momentum: float | None = None,
    kfac_norm_constraint: float | None = None,
    kfac_l2_reg: float | None = None,
    kfac_cov_ema_decay: float | None = None,
    kfac_invert_every: int | None = None,
    kfac_register_only_generic: bool | None = None,
    running_stats: ExternalStatePenaltyRunningStats | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
    ratio_clip_width: float | None = 10.0,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = 30.0,
    local_energy_clip_width: float | None = 5.0,
    local_energy_exclude_width: float = float("inf"),
    overlap_scale_by: str | None = "max_gap_std",
    overlap_ewma_decay: float | None = None,
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    state_ordering: str = "index",
    gradient_mode: str = "paper_tangent",
    max_grad_l2_norm: float | None = None,
    max_update_l2_norm: float | None = None,
    param_share_keys: tuple[str, ...] | list[str] | None = None,
    candidate_check_period: int = 1,
    reject_nonfinite_update: bool = True,
    block_until_ready: bool = False,
) -> ExternalStateDriverResult:
    """Run sampler-integrated external-state penalty optimization."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if sampler_burn_in < 0:
        raise ValueError("sampler_burn_in must be non-negative")
    if sampler_steps_per_iteration < 0:
        raise ValueError("sampler_steps_per_iteration must be non-negative")
    if sampler_burn_in + sampler_steps_per_iteration < 1:
        raise ValueError("sampler must perform at least one step")
    if sampler_proposal_width <= 0.0:
        raise ValueError("sampler_proposal_width must be positive")
    if candidate_check_period < 1:
        raise ValueError("candidate_check_period must be at least 1")

    params, _ = merge_external_state_params(adapter, state_params, param_share_keys)
    current_samples = samples
    key = sampler_key
    opt_state = merge_external_optimizer_state(
        adapter,
        optimizer_state,
        param_share_keys,
    )
    stats = running_stats
    scale_energy = None if stats is None else stats.state_energy_ewma
    scale_std = None if stats is None else stats.state_energy_std_ewma
    initial_terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        current_samples,
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
        overlap_scale_state_energy=scale_energy,
        overlap_scale_state_energy_std=scale_std,
    )
    stats = update_external_state_penalty_running_stats(
        adapter,
        stats,
        initial_terms,
        decay=overlap_ewma_decay,
    )
    history = []
    for iteration in range(iterations):
        sampling_steps = sampler_steps_per_iteration
        if iteration == 0:
            sampling_steps += sampler_burn_in
        key, sample_key = adapter.modules.jax.random.split(key)
        current_samples, sampler_diagnostics = metropolis_update_state_samples(
            adapter,
            params,
            current_samples,
            key=sample_key,
            iteration=iteration,
            steps=sampling_steps,
            proposal_width=sampler_proposal_width,
            wrap=sampler_wrap,
        )
        params, opt_state, stats, update_diagnostics = external_state_penalty_optimizer_step(
            adapter,
            params,
            current_samples,
            step=iteration,
            learning_rate=learning_rate,
            penalty_alpha=penalty_alpha,
            local_energy=local_energy,
            energy_weights=energy_weights,
            optimizer_name=optimizer_name,
            optimizer_state=opt_state,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            lamb_eps=lamb_eps,
            kfac_damping=kfac_damping,
            kfac_momentum=kfac_momentum,
            kfac_norm_constraint=kfac_norm_constraint,
            kfac_l2_reg=kfac_l2_reg,
            kfac_cov_ema_decay=kfac_cov_ema_decay,
            kfac_invert_every=kfac_invert_every,
            kfac_register_only_generic=kfac_register_only_generic,
            running_stats=stats,
            collapse_threshold=collapse_threshold,
            clip_upper=clip_upper,
            ratio_clip_width=ratio_clip_width,
            ratio_exclude_width=ratio_exclude_width,
            max_logabs_ratio=max_logabs_ratio,
            local_energy_clip_width=local_energy_clip_width,
            local_energy_exclude_width=local_energy_exclude_width,
            overlap_scale_by=overlap_scale_by,
            overlap_ewma_decay=overlap_ewma_decay,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
            state_ordering=state_ordering,
            gradient_mode=gradient_mode,
            max_grad_l2_norm=max_grad_l2_norm,
            max_update_l2_norm=max_update_l2_norm,
            param_share_keys=param_share_keys,
            candidate_check_period=candidate_check_period,
            reject_nonfinite_update=reject_nonfinite_update,
            block_until_ready=block_until_ready,
        )
        history.append(
            ExternalStateDriverStepDiagnostics(
                iteration=iteration,
                sampler=sampler_diagnostics,
                update=update_diagnostics,
            )
        )

    scale_energy = None if stats is None else stats.state_energy_ewma
    scale_std = None if stats is None else stats.state_energy_std_ewma
    final_terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        current_samples,
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
        overlap_scale_state_energy=scale_energy,
        overlap_scale_state_energy_std=scale_std,
    )
    if block_until_ready:
        params = block_tree_until_ready(adapter, params)
        current_samples = block_tree_until_ready(adapter, current_samples)
        key = block_tree_until_ready(adapter, key)
        initial_terms = block_tree_until_ready(adapter, initial_terms)
        final_terms = block_tree_until_ready(adapter, final_terms)
        opt_state = block_tree_until_ready(adapter, opt_state)
        stats = block_tree_until_ready(adapter, stats)
    return ExternalStateDriverResult(
        state_params=params,
        samples=current_samples,
        sampler_key=key,
        initial_terms=initial_terms,
        final_terms=final_terms,
        history=tuple(history),
        optimizer_state=opt_state,
        running_stats=stats,
    )


def metropolis_update_state_samples(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    key: Any,
    iteration: int,
    steps: int,
    proposal_width: float,
    wrap: bool = True,
) -> tuple[FermiNetPBCStateSamples, ExternalStateMetropolisDiagnostics]:
    """Run per-state random-walk Metropolis updates for external samples."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if proposal_width <= 0.0:
        raise ValueError("proposal_width must be positive")
    jax = adapter.modules.jax
    jnp = adapter.modules.jnp
    positions = jnp.asarray(samples.positions)
    total_accept = jnp.asarray(0.0)
    total_trials = jnp.asarray(0.0)
    keys = jax.random.split(key, len(state_params))
    for state_idx, params in enumerate(state_params):
        state_positions = positions[state_idx]
        spins, atoms, charges = (
            jnp.asarray(samples.spins)[state_idx],
            jnp.asarray(samples.atoms)[state_idx],
            jnp.asarray(samples.charges)[state_idx],
        )
        _, current_logabs = adapter.batched_signed_network(
            params,
            state_positions,
            spins,
            atoms,
            charges,
        )
        state_key = keys[state_idx]
        for _ in range(steps):
            state_key, proposal_key, accept_key = jax.random.split(state_key, 3)
            proposal = state_positions + proposal_width * jax.random.normal(
                proposal_key,
                state_positions.shape,
            )
            if wrap:
                proposal = wrap_flat_positions_in_lattice(adapter, proposal)
            _, proposal_logabs = adapter.batched_signed_network(
                params,
                proposal,
                spins,
                atoms,
                charges,
            )
            log_accept = 2.0 * jnp.real(proposal_logabs - current_logabs)
            log_uniform = jnp.log(
                jax.random.uniform(
                    accept_key,
                    log_accept.shape,
                    minval=1.0e-12,
                    maxval=1.0,
                )
            )
            accepted = log_uniform < jnp.minimum(log_accept, 0.0)
            state_positions = jnp.where(accepted[:, None], proposal, state_positions)
            current_logabs = jnp.where(accepted, proposal_logabs, current_logabs)
            total_accept = total_accept + jnp.sum(accepted)
            total_trials = total_trials + accepted.size
        positions = positions.at[state_idx].set(state_positions)
    acceptance = jnp.where(total_trials > 0, total_accept / total_trials, 0.0)
    return (
        broadcast_state_samples(
            positions=positions,
            spins=samples.spins,
            atoms=samples.atoms,
            charges=samples.charges,
        ),
        ExternalStateMetropolisDiagnostics(
            iteration=iteration,
            steps=steps,
            acceptance=acceptance,
        ),
    )


def wrap_flat_positions_in_lattice(
    adapter: FermiNetPBCExternalStateAdapter,
    positions: Any,
) -> Any:
    """Wrap flattened Cartesian electron positions back into the PBC cell."""

    jnp = adapter.modules.jnp
    lattice = jnp.asarray(adapter.cfg.system.make_local_energy_kwargs["lattice"])
    inv_lattice = jnp.linalg.inv(lattice)
    ndim = int(lattice.shape[0])
    nelec = sum(adapter.nspins)
    original_shape = positions.shape
    cart = jnp.asarray(positions).reshape(original_shape[:-1] + (nelec, ndim))
    frac = jnp.einsum("...j,ij->...i", cart, inv_lattice)
    wrapped_frac = frac - jnp.floor(frac)
    wrapped_cart = jnp.einsum("...i,ji->...j", wrapped_frac, lattice)
    return wrapped_cart.reshape(original_shape)


def save_external_state_driver_checkpoint(
    path: str | Path,
    *,
    result: ExternalStateDriverResult,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist driver params, samples, sampler key, and lightweight metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "state_params": _tree_to_numpy(result.state_params),
        "samples": _tree_to_numpy(result.samples),
        "sampler_key": _tree_to_numpy(result.sampler_key),
        "optimizer_state": _tree_to_numpy(result.optimizer_state),
        "running_stats": _tree_to_numpy(result.running_stats),
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return checkpoint_path


def load_external_state_driver_checkpoint(
    path: str | Path,
    *,
    adapter: FermiNetPBCExternalStateAdapter | None = None,
) -> dict[str, Any]:
    """Load a checkpoint payload, optionally converting arrays to adapter JAX arrays."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if adapter is None:
        return payload
    return {
        "metadata": payload["metadata"],
        "state_params": _tree_from_numpy(adapter, payload["state_params"]),
        "samples": _tree_from_numpy(adapter, payload["samples"]),
        "sampler_key": _tree_from_numpy(adapter, payload["sampler_key"]),
        "optimizer_state": _tree_from_numpy(adapter, payload.get("optimizer_state")),
        "running_stats": _tree_from_numpy(adapter, payload.get("running_stats")),
    }


def _tree_to_numpy(tree: Any) -> Any:
    if isinstance(tree, ExternalStateOptimizerState):
        return {
            "__solidnes_type__": "ExternalStateOptimizerState",
            "name": tree.name,
            "step": tree.step,
            "first_moment": _tree_to_numpy(tree.first_moment),
            "second_moment": _tree_to_numpy(tree.second_moment),
            "native_state": _tree_to_numpy(tree.native_state),
            "native_state_metadata": _tree_to_numpy(tree.native_state_metadata),
        }
    if isinstance(tree, ExternalStatePenaltyRunningStats):
        return {
            "__solidnes_type__": "ExternalStatePenaltyRunningStats",
            "step": tree.step,
            "state_energy_ewma": _tree_to_numpy(tree.state_energy_ewma),
            "state_energy_std_ewma": _tree_to_numpy(tree.state_energy_std_ewma),
            "decay": tree.decay,
        }
    if isinstance(tree, FermiNetPBCStateSamples):
        return {
            "positions": np.asarray(tree.positions),
            "spins": np.asarray(tree.spins),
            "atoms": np.asarray(tree.atoms),
            "charges": np.asarray(tree.charges),
        }
    if isinstance(tree, dict):
        return {key: _tree_to_numpy(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_to_numpy(value) for value in tree)
    if isinstance(tree, list):
        return [_tree_to_numpy(value) for value in tree]
    if hasattr(tree, "shape"):
        return np.asarray(tree)
    return tree


def _tree_from_numpy(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    if tree is None:
        return None
    if isinstance(tree, dict) and tree.get("__solidnes_type__") == "ExternalStateOptimizerState":
        return ExternalStateOptimizerState(
            name=tree["name"],
            step=int(tree["step"]),
            first_moment=_tree_from_numpy(adapter, tree.get("first_moment")),
            second_moment=_tree_from_numpy(adapter, tree.get("second_moment")),
            native_state=_tree_from_numpy(adapter, tree.get("native_state")),
            native_state_metadata=_tree_from_numpy(
                adapter,
                tree.get("native_state_metadata"),
            ),
        )
    if isinstance(tree, dict) and tree.get("__solidnes_type__") == "ExternalStatePenaltyRunningStats":
        return ExternalStatePenaltyRunningStats(
            step=int(tree["step"]),
            state_energy_ewma=_tree_from_numpy(adapter, tree["state_energy_ewma"]),
            state_energy_std_ewma=_tree_from_numpy(
                adapter,
                tree["state_energy_std_ewma"],
            ),
            decay=float(tree["decay"]),
        )
    if isinstance(tree, dict) and set(tree) == {"positions", "spins", "atoms", "charges"}:
        return broadcast_state_samples(
            positions=adapter.modules.jnp.asarray(tree["positions"]),
            spins=adapter.modules.jnp.asarray(tree["spins"]),
            atoms=adapter.modules.jnp.asarray(tree["atoms"]),
            charges=adapter.modules.jnp.asarray(tree["charges"]),
        )
    if isinstance(tree, dict):
        return {key: _tree_from_numpy(adapter, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_from_numpy(adapter, value) for value in tree)
    if isinstance(tree, list):
        return [_tree_from_numpy(adapter, value) for value in tree]
    if hasattr(tree, "shape"):
        return adapter.modules.jnp.asarray(tree)
    return tree


__all__ = [
    "ExternalStateDriverResult",
    "ExternalStateDriverStepDiagnostics",
    "ExternalStateMetropolisDiagnostics",
    "load_external_state_driver_checkpoint",
    "metropolis_update_state_samples",
    "run_external_state_penalty_driver",
    "save_external_state_driver_checkpoint",
    "wrap_flat_positions_in_lattice",
]
