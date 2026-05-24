"""Training-loop helpers for externally managed FermiNet PBC states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solidnes.excited_states.ferminet_pbc_adapter import (
    FermiNetPBCExternalStateAdapter,
)
from solidnes.excited_states.ferminet_pbc_adapter import (
    apply_external_state_sgd_step,
)
from solidnes.excited_states.ferminet_pbc_adapter import (
    evaluate_ferminet_pbc_penalty_terms,
)
from solidnes.excited_states.ferminet_pbc_adapter import (
    value_and_grad_ferminet_pbc_penalty_objective,
)
from solidnes.excited_states.ferminet_pbc_scaffold import BatchedLocalEnergy
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples


@dataclass(frozen=True)
class ExternalStatePenaltyStepDiagnostics:
    """Diagnostics for one external-state penalty-objective update."""

    step: int
    penalty_objective: Any
    weighted_state_energy: Any
    state_energy: Any
    offdiag_squared_overlap: Any
    overlap_penalty: Any
    max_abs_offdiag_overlap: Any
    collapse_flag: Any
    grad_l2_norm: Any
    param_delta_l2_norm: Any


@dataclass(frozen=True)
class ExternalStatePenaltyTrainingResult:
    """Result of a fixed-sample external-state penalty optimization loop."""

    state_params: tuple[Any, ...]
    initial_terms: dict[str, Any]
    final_terms: dict[str, Any]
    history: tuple[ExternalStatePenaltyStepDiagnostics, ...]


def run_external_state_penalty_sgd(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    steps: int,
    learning_rate: float,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
    block_until_ready: bool = False,
) -> ExternalStatePenaltyTrainingResult:
    """Run fixed-sample SGD over externally managed FermiNet PBC states.

    This helper deliberately owns only the penalty objective and state-parameter
    update.  MCMC sampling, adaptive schedules, checkpointing, and optimizer
    state are intentionally deferred to a later production driver.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    params = state_params
    initial_terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        samples,
        penalty_alpha=penalty_alpha,
        local_energy=local_energy,
        energy_weights=energy_weights,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )
    history = []
    for step in range(steps):
        params, diagnostics = external_state_penalty_sgd_step(
            adapter,
            params,
            samples,
            step=step,
            learning_rate=learning_rate,
            penalty_alpha=penalty_alpha,
            local_energy=local_energy,
            energy_weights=energy_weights,
            collapse_threshold=collapse_threshold,
            clip_upper=clip_upper,
            block_until_ready=block_until_ready,
        )
        history.append(diagnostics)

    final_terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        params,
        samples,
        penalty_alpha=penalty_alpha,
        local_energy=local_energy,
        energy_weights=energy_weights,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )
    if block_until_ready:
        initial_terms = block_tree_until_ready(adapter, initial_terms)
        final_terms = block_tree_until_ready(adapter, final_terms)
    return ExternalStatePenaltyTrainingResult(
        state_params=params,
        initial_terms=initial_terms,
        final_terms=final_terms,
        history=tuple(history),
    )


def external_state_penalty_sgd_step(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    step: int,
    learning_rate: float,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None = None,
    energy_weights: Any | None = None,
    collapse_threshold: float = 0.95,
    clip_upper: bool = False,
    block_until_ready: bool = False,
) -> tuple[tuple[Any, ...], ExternalStatePenaltyStepDiagnostics]:
    """Apply one fixed-sample SGD step and return pre-update diagnostics."""

    terms = evaluate_ferminet_pbc_penalty_terms(
        adapter,
        state_params,
        samples,
        penalty_alpha=penalty_alpha,
        local_energy=local_energy,
        energy_weights=energy_weights,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )
    objective, grads = value_and_grad_ferminet_pbc_penalty_objective(
        adapter,
        state_params,
        samples,
        penalty_alpha=penalty_alpha,
        local_energy=local_energy,
        energy_weights=energy_weights,
        collapse_threshold=collapse_threshold,
        clip_upper=clip_upper,
    )
    grad_norm = tree_l2_norm(adapter, grads)
    new_params = apply_external_state_sgd_step(
        adapter,
        state_params,
        grads,
        learning_rate=learning_rate,
    )
    delta = adapter.modules.jax.tree_util.tree_map(
        lambda old, new: new - old,
        state_params,
        new_params,
    )
    delta_norm = tree_l2_norm(adapter, delta)
    if block_until_ready:
        objective = block_tree_until_ready(adapter, objective)
        terms = block_tree_until_ready(adapter, terms)
        grad_norm = block_tree_until_ready(adapter, grad_norm)
        delta_norm = block_tree_until_ready(adapter, delta_norm)
        new_params = block_tree_until_ready(adapter, new_params)

    diagnostics = ExternalStatePenaltyStepDiagnostics(
        step=step,
        penalty_objective=objective,
        weighted_state_energy=terms["weighted_state_energy"],
        state_energy=terms["state_energy"],
        offdiag_squared_overlap=terms["offdiag_squared_overlap"],
        overlap_penalty=terms["overlap_penalty"],
        max_abs_offdiag_overlap=terms["max_abs_offdiag_overlap"],
        collapse_flag=terms["collapse_flag"],
        grad_l2_norm=grad_norm,
        param_delta_l2_norm=delta_norm,
    )
    return new_params, diagnostics


def tree_l2_norm(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Return the real-valued L2 norm of a JAX pytree."""

    jnp = adapter.modules.jnp
    leaves = adapter.modules.jax.tree_util.tree_leaves(tree)
    total = jnp.asarray(0.0)
    for leaf in leaves:
        array = jnp.asarray(leaf)
        total = total + jnp.sum(jnp.real(array * jnp.conj(array)))
    return jnp.sqrt(total)


def block_tree_until_ready(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Block on all JAX arrays in a pytree and return the same tree shape."""

    def block(value: Any) -> Any:
        if hasattr(value, "block_until_ready"):
            return value.block_until_ready()
        return value

    return adapter.modules.jax.tree_util.tree_map(block, tree)


__all__ = [
    "ExternalStatePenaltyStepDiagnostics",
    "ExternalStatePenaltyTrainingResult",
    "block_tree_until_ready",
    "external_state_penalty_sgd_step",
    "run_external_state_penalty_sgd",
    "tree_l2_norm",
]
