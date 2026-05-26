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
    ferminet_pbc_penalty_training_objective,
)
from solidnes.excited_states.ferminet_pbc_adapter import (
    value_and_grad_ferminet_pbc_penalty_objective,
)
from solidnes.excited_states.ferminet_pbc_scaffold import BatchedLocalEnergy
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples


_EXTERNAL_STATE_KFAC_OPTIMIZER_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class ExternalStateOptimizerState:
    """Optimizer state for externally managed state-parameter pytrees."""

    name: str
    step: int = 0
    first_moment: Any | None = None
    second_moment: Any | None = None
    native_state: Any | None = None
    native_state_metadata: Any | None = None


@dataclass(frozen=True)
class ExternalStatePenaltyRunningStats:
    """Running statistics used by paper-style overlap-gradient scaling."""

    step: int
    state_energy_ewma: Any
    state_energy_std_ewma: Any
    decay: float


@dataclass(frozen=True)
class ExternalStatePenaltyStepDiagnostics:
    """Diagnostics for one external-state penalty-objective update."""

    step: int
    penalty_objective: Any
    gradient_objective: Any
    weighted_state_energy: Any
    state_energy: Any
    offdiag_squared_overlap: Any
    scaled_offdiag_squared_overlap: Any
    overlap_penalty: Any
    overlap_gradient_scale: Any
    max_abs_offdiag_overlap: Any
    collapse_flag: Any
    state_energy_std: Any
    grad_l2_norm: Any
    param_delta_l2_norm: Any
    gradient_objective_finite: Any
    gradient_finite: Any
    update_finite: Any
    candidate_terms_finite: Any
    update_accepted: Any
    optimizer_update_l2_norm: Any | None = None
    share_projection_l2_norm: Any | None = None
    optimizer_name: str = "sgd"
    optimizer_step: Any | None = None
    candidate_check_performed: Any | None = None
    shared_param_paths: tuple[str, ...] = ()
    state_energy_ewma: Any | None = None
    state_energy_std_ewma: Any | None = None


@dataclass(frozen=True)
class ExternalStatePenaltyTrainingResult:
    """Result of a fixed-sample external-state penalty optimization loop."""

    state_params: tuple[Any, ...]
    initial_terms: dict[str, Any]
    final_terms: dict[str, Any]
    history: tuple[ExternalStatePenaltyStepDiagnostics, ...]
    optimizer_state: ExternalStateOptimizerState | None = None
    running_stats: ExternalStatePenaltyRunningStats | None = None


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
    max_grad_l2_norm: float | None = None,
    max_update_l2_norm: float | None = None,
    param_share_keys: tuple[str, ...] | list[str] | None = None,
    candidate_check_period: int = 1,
    reject_nonfinite_update: bool = True,
    block_until_ready: bool = False,
) -> ExternalStatePenaltyTrainingResult:
    """Run fixed-sample SGD over externally managed FermiNet PBC states.

    This helper deliberately owns only the fixed-sample penalty objective and
    state-parameter update.  MCMC sampling and checkpointing live in the
    production driver, while optimizer state is returned here for reuse there.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    params, _ = merge_external_state_params(adapter, state_params, param_share_keys)
    opt_state = optimizer_state
    opt_state = merge_external_optimizer_state(adapter, opt_state, param_share_keys)
    running_stats = None
    initial_terms = evaluate_ferminet_pbc_penalty_terms(
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
    )
    running_stats = update_external_state_penalty_running_stats(
        adapter,
        running_stats,
        initial_terms,
        decay=overlap_ewma_decay,
    )
    history = []
    for step in range(steps):
        params, opt_state, running_stats, diagnostics = (
            external_state_penalty_optimizer_step(
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
                ratio_clip_width=ratio_clip_width,
                ratio_exclude_width=ratio_exclude_width,
                max_logabs_ratio=max_logabs_ratio,
                local_energy_clip_width=local_energy_clip_width,
                local_energy_exclude_width=local_energy_exclude_width,
                overlap_scale_by=overlap_scale_by,
                overlap_ewma_decay=overlap_ewma_decay,
                running_stats=running_stats,
                min_gap_scale_factor=min_gap_scale_factor,
                max_scale_factor=max_scale_factor,
                state_ordering=state_ordering,
                gradient_mode=gradient_mode,
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
                max_grad_l2_norm=max_grad_l2_norm,
                max_update_l2_norm=max_update_l2_norm,
                param_share_keys=param_share_keys,
                candidate_check_period=candidate_check_period,
                reject_nonfinite_update=reject_nonfinite_update,
                block_until_ready=block_until_ready,
            )
        )
        history.append(diagnostics)

    scale_energy, scale_std = _running_scale_inputs(running_stats)
    final_terms = evaluate_ferminet_pbc_penalty_terms(
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
        overlap_scale_state_energy=scale_energy,
        overlap_scale_state_energy_std=scale_std,
    )
    if block_until_ready:
        initial_terms = block_tree_until_ready(adapter, initial_terms)
        final_terms = block_tree_until_ready(adapter, final_terms)
        opt_state = block_tree_until_ready(adapter, opt_state)
        running_stats = block_tree_until_ready(adapter, running_stats)
    return ExternalStatePenaltyTrainingResult(
        state_params=params,
        initial_terms=initial_terms,
        final_terms=final_terms,
        history=tuple(history),
        optimizer_state=opt_state,
        running_stats=running_stats,
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
    max_grad_l2_norm: float | None = None,
    reject_nonfinite_update: bool = True,
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
        ratio_clip_width=ratio_clip_width,
        ratio_exclude_width=ratio_exclude_width,
        max_logabs_ratio=max_logabs_ratio,
        overlap_scale_by=overlap_scale_by,
        min_gap_scale_factor=min_gap_scale_factor,
        max_scale_factor=max_scale_factor,
    )
    gradient_objective, grads = value_and_grad_ferminet_pbc_penalty_objective(
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
        local_energy_clip_width=local_energy_clip_width,
        local_energy_exclude_width=local_energy_exclude_width,
        overlap_scale_by=overlap_scale_by,
        min_gap_scale_factor=min_gap_scale_factor,
        max_scale_factor=max_scale_factor,
        state_ordering=state_ordering,
        gradient_mode=gradient_mode,
        precomputed_terms=terms if gradient_mode == "paper_tangent" else None,
    )
    grad_norm = tree_l2_norm(adapter, grads)
    if max_grad_l2_norm is not None:
        grads = clip_tree_by_global_norm(
            adapter,
            grads,
            grad_norm=grad_norm,
            max_norm=max_grad_l2_norm,
        )
        grad_norm = tree_l2_norm(adapter, grads)
    gradient_objective_finite = scalar_all_finite(adapter, gradient_objective)
    gradient_finite = (
        gradient_objective_finite
        & tree_all_finite(adapter, grads)
        & adapter.modules.jnp.isfinite(grad_norm)
    )
    update_grads = sanitize_tree_nonfinite(adapter, grads)
    new_params = apply_external_state_sgd_step(
        adapter,
        state_params,
        update_grads,
        learning_rate=learning_rate,
    )
    delta = adapter.modules.jax.tree_util.tree_map(
        lambda old, new: new - old,
        state_params,
        new_params,
    )
    update_finite = tree_all_finite(adapter, new_params) & tree_all_finite(adapter, delta)
    if reject_nonfinite_update:
        candidate_terms = evaluate_ferminet_pbc_penalty_terms(
            adapter,
            new_params,
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
        )
        candidate_terms_finite = penalty_terms_all_finite(adapter, candidate_terms)
        update_accepted = gradient_finite & update_finite & candidate_terms_finite
        new_params = adapter.modules.jax.tree_util.tree_map(
            lambda old, new: adapter.modules.jnp.where(update_accepted, new, old),
            state_params,
            new_params,
        )
        delta = adapter.modules.jax.tree_util.tree_map(
            lambda old, new: new - old,
            state_params,
            new_params,
        )
    else:
        candidate_terms_finite = adapter.modules.jnp.asarray(True)
        update_accepted = adapter.modules.jnp.asarray(True)
    delta_norm = tree_l2_norm(adapter, delta)
    if block_until_ready:
        gradient_objective = block_tree_until_ready(adapter, gradient_objective)
        terms = block_tree_until_ready(adapter, terms)
        grad_norm = block_tree_until_ready(adapter, grad_norm)
        delta_norm = block_tree_until_ready(adapter, delta_norm)
        gradient_objective_finite = block_tree_until_ready(
            adapter,
            gradient_objective_finite,
        )
        gradient_finite = block_tree_until_ready(adapter, gradient_finite)
        update_finite = block_tree_until_ready(adapter, update_finite)
        candidate_terms_finite = block_tree_until_ready(adapter, candidate_terms_finite)
        update_accepted = block_tree_until_ready(adapter, update_accepted)
        new_params = block_tree_until_ready(adapter, new_params)

    diagnostics = ExternalStatePenaltyStepDiagnostics(
        step=step,
        penalty_objective=terms["penalty_objective"],
        gradient_objective=gradient_objective,
        weighted_state_energy=terms["weighted_state_energy"],
        state_energy=terms["state_energy"],
        offdiag_squared_overlap=terms["offdiag_squared_overlap"],
        scaled_offdiag_squared_overlap=terms["scaled_offdiag_squared_overlap"],
        overlap_penalty=terms["overlap_penalty"],
        overlap_gradient_scale=terms["overlap_gradient_scale"],
        max_abs_offdiag_overlap=terms["max_abs_offdiag_overlap"],
        collapse_flag=terms["collapse_flag"],
        state_energy_std=terms["state_energy_std"],
        grad_l2_norm=grad_norm,
        param_delta_l2_norm=delta_norm,
        optimizer_update_l2_norm=delta_norm,
        share_projection_l2_norm=adapter.modules.jnp.asarray(0.0),
        gradient_objective_finite=gradient_objective_finite,
        gradient_finite=gradient_finite,
        update_finite=update_finite,
        candidate_terms_finite=candidate_terms_finite,
        update_accepted=update_accepted,
    )
    return new_params, diagnostics


def external_state_penalty_optimizer_step(
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
    ratio_clip_width: float | None = 10.0,
    ratio_exclude_width: float = float("inf"),
    max_logabs_ratio: float | None = 30.0,
    local_energy_clip_width: float | None = 5.0,
    local_energy_exclude_width: float = float("inf"),
    overlap_scale_by: str | None = "max_gap_std",
    overlap_ewma_decay: float | None = None,
    running_stats: ExternalStatePenaltyRunningStats | None = None,
    min_gap_scale_factor: float = 0.001,
    max_scale_factor: float = 5.0,
    state_ordering: str = "index",
    gradient_mode: str = "paper_tangent",
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
    max_grad_l2_norm: float | None = None,
    max_update_l2_norm: float | None = None,
    param_share_keys: tuple[str, ...] | list[str] | None = None,
    candidate_check_period: int = 1,
    reject_nonfinite_update: bool = True,
    block_until_ready: bool = False,
) -> tuple[
    tuple[Any, ...],
    ExternalStateOptimizerState | None,
    ExternalStatePenaltyRunningStats | None,
    ExternalStatePenaltyStepDiagnostics,
]:
    """Apply one fixed-sample optimizer step and return pre-update diagnostics."""

    if candidate_check_period < 1:
        raise ValueError("candidate_check_period must be at least 1")

    jnp = adapter.modules.jnp
    normalized_optimizer_name = _normalize_optimizer_name(optimizer_name)
    optimizer_uses_kfac = normalized_optimizer_name == "kfac"
    scale_energy, scale_std = _running_scale_inputs(running_stats)
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
        overlap_scale_state_energy=scale_energy,
        overlap_scale_state_energy_std=scale_std,
    )
    next_running_stats = update_external_state_penalty_running_stats(
        adapter,
        running_stats,
        terms,
        decay=overlap_ewma_decay,
    )
    if optimizer_uses_kfac:
        gradient_objective = terms["penalty_objective"]
        grad_norm = jnp.asarray(0.0)
        gradient_objective_finite = scalar_all_finite(adapter, gradient_objective)
        gradient_finite = gradient_objective_finite & penalty_terms_all_finite(
            adapter,
            terms,
        )
        update_grads = None
    else:
        gradient_objective, grads = value_and_grad_ferminet_pbc_penalty_objective(
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
            local_energy_clip_width=local_energy_clip_width,
            local_energy_exclude_width=local_energy_exclude_width,
            overlap_scale_by=overlap_scale_by,
            min_gap_scale_factor=min_gap_scale_factor,
            max_scale_factor=max_scale_factor,
            state_ordering=state_ordering,
            gradient_mode=gradient_mode,
            precomputed_terms=terms if gradient_mode == "paper_tangent" else None,
            overlap_scale_state_energy=scale_energy,
            overlap_scale_state_energy_std=scale_std,
        )
        grad_norm = tree_l2_norm(adapter, grads)
        if max_grad_l2_norm is not None:
            grads = clip_tree_by_global_norm(
                adapter,
                grads,
                grad_norm=grad_norm,
                max_norm=max_grad_l2_norm,
            )
            grad_norm = tree_l2_norm(adapter, grads)
        gradient_objective_finite = scalar_all_finite(adapter, gradient_objective)
        gradient_finite = (
            gradient_objective_finite
            & tree_all_finite(adapter, grads)
            & jnp.isfinite(grad_norm)
        )
        update_grads = sanitize_tree_nonfinite(adapter, grads)
    if optimizer_uses_kfac:
        updated_params, next_optimizer_state, optimizer_updates = (
            apply_external_state_kfac_step(
                adapter,
                state_params,
                samples,
                optimizer_state=optimizer_state,
                step=step,
                learning_rate=learning_rate,
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
                gradient_mode=gradient_mode,
                overlap_scale_state_energy=scale_energy,
                overlap_scale_state_energy_std=scale_std,
                precomputed_terms=terms,
                max_update_l2_norm=max_update_l2_norm,
                kfac_damping=kfac_damping,
                kfac_momentum=kfac_momentum,
                kfac_norm_constraint=kfac_norm_constraint,
                kfac_l2_reg=kfac_l2_reg,
                kfac_cov_ema_decay=kfac_cov_ema_decay,
                kfac_invert_every=kfac_invert_every,
                kfac_register_only_generic=kfac_register_only_generic,
            )
        )
    else:
        updated_params, next_optimizer_state, optimizer_updates = (
            apply_external_state_optimizer_step(
                adapter,
                state_params,
                update_grads,
                optimizer_state=optimizer_state,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                adam_b1=adam_b1,
                adam_b2=adam_b2,
                adam_eps=adam_eps,
                weight_decay=weight_decay,
                lamb_eps=lamb_eps,
                max_update_l2_norm=max_update_l2_norm,
            )
        )
    optimizer_update_norm = (
        jnp.asarray(0.0)
        if optimizer_updates is None
        else tree_l2_norm(adapter, optimizer_updates)
    )
    if optimizer_uses_kfac:
        grad_norm = optimizer_update_norm
        gradient_finite = gradient_finite & jnp.isfinite(grad_norm)
    new_params, shared_param_paths = merge_external_state_params(
        adapter,
        updated_params,
        param_share_keys,
    )
    share_projection = adapter.modules.jax.tree_util.tree_map(
        lambda before, after: after - before,
        updated_params,
        new_params,
    )
    share_projection_norm = tree_l2_norm(adapter, share_projection)
    next_optimizer_state = merge_external_optimizer_state(
        adapter,
        next_optimizer_state,
        param_share_keys,
    )
    delta = adapter.modules.jax.tree_util.tree_map(
        lambda old, new: new - old,
        state_params,
        new_params,
    )
    update_finite = tree_all_finite(adapter, new_params) & tree_all_finite(adapter, delta)
    candidate_check_performed = jnp.asarray(
        reject_nonfinite_update and (step % candidate_check_period == 0)
    )
    if reject_nonfinite_update:
        if step % candidate_check_period == 0:
            candidate_terms = evaluate_ferminet_pbc_penalty_terms(
                adapter,
                new_params,
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
                overlap_scale_state_energy=scale_energy,
                overlap_scale_state_energy_std=scale_std,
            )
            candidate_terms_finite = penalty_terms_all_finite(adapter, candidate_terms)
        else:
            candidate_terms_finite = jnp.asarray(True)
        update_accepted = gradient_finite & update_finite & candidate_terms_finite
        new_params = adapter.modules.jax.tree_util.tree_map(
            lambda old, new: adapter.modules.jnp.where(update_accepted, new, old),
            state_params,
            new_params,
        )
        next_optimizer_state = _select_optimizer_state(
            adapter,
            update_accepted,
            optimizer_state,
            next_optimizer_state,
        )
        delta = adapter.modules.jax.tree_util.tree_map(
            lambda old, new: new - old,
            state_params,
            new_params,
        )
    else:
        candidate_terms_finite = jnp.asarray(True)
        update_accepted = jnp.asarray(True)
    delta_norm = tree_l2_norm(adapter, delta)
    if block_until_ready:
        gradient_objective = block_tree_until_ready(adapter, gradient_objective)
        terms = block_tree_until_ready(adapter, terms)
        grad_norm = block_tree_until_ready(adapter, grad_norm)
        delta_norm = block_tree_until_ready(adapter, delta_norm)
        optimizer_update_norm = block_tree_until_ready(adapter, optimizer_update_norm)
        share_projection_norm = block_tree_until_ready(adapter, share_projection_norm)
        gradient_objective_finite = block_tree_until_ready(
            adapter,
            gradient_objective_finite,
        )
        gradient_finite = block_tree_until_ready(adapter, gradient_finite)
        update_finite = block_tree_until_ready(adapter, update_finite)
        candidate_terms_finite = block_tree_until_ready(adapter, candidate_terms_finite)
        candidate_check_performed = block_tree_until_ready(
            adapter,
            candidate_check_performed,
        )
        update_accepted = block_tree_until_ready(adapter, update_accepted)
        new_params = block_tree_until_ready(adapter, new_params)
        next_optimizer_state = block_tree_until_ready(adapter, next_optimizer_state)
        next_running_stats = block_tree_until_ready(adapter, next_running_stats)

    diagnostics = ExternalStatePenaltyStepDiagnostics(
        step=step,
        penalty_objective=terms["penalty_objective"],
        gradient_objective=gradient_objective,
        weighted_state_energy=terms["weighted_state_energy"],
        state_energy=terms["state_energy"],
        offdiag_squared_overlap=terms["offdiag_squared_overlap"],
        scaled_offdiag_squared_overlap=terms["scaled_offdiag_squared_overlap"],
        overlap_penalty=terms["overlap_penalty"],
        overlap_gradient_scale=terms["overlap_gradient_scale"],
        max_abs_offdiag_overlap=terms["max_abs_offdiag_overlap"],
        collapse_flag=terms["collapse_flag"],
        state_energy_std=terms["state_energy_std"],
        grad_l2_norm=grad_norm,
        param_delta_l2_norm=delta_norm,
        optimizer_update_l2_norm=optimizer_update_norm,
        share_projection_l2_norm=share_projection_norm,
        gradient_objective_finite=gradient_objective_finite,
        gradient_finite=gradient_finite,
        update_finite=update_finite,
        candidate_terms_finite=candidate_terms_finite,
        update_accepted=update_accepted,
        optimizer_name=_normalize_optimizer_name(optimizer_name),
        optimizer_step=None if next_optimizer_state is None else next_optimizer_state.step,
        candidate_check_performed=candidate_check_performed,
        shared_param_paths=shared_param_paths,
        state_energy_ewma=None
        if next_running_stats is None
        else next_running_stats.state_energy_ewma,
        state_energy_std_ewma=None
        if next_running_stats is None
        else next_running_stats.state_energy_std_ewma,
    )
    return new_params, next_optimizer_state, next_running_stats, diagnostics


def init_external_state_optimizer(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    *,
    optimizer_name: str,
) -> ExternalStateOptimizerState:
    """Initialize a lightweight external-state optimizer state."""

    name = _normalize_optimizer_name(optimizer_name)
    if name in ("sgd", "none"):
        return ExternalStateOptimizerState(name=name, step=0)
    if name in ("adam", "lamb"):
        zeros = adapter.modules.jax.tree_util.tree_map(
            lambda leaf: adapter.modules.jnp.zeros_like(leaf),
            state_params,
        )
        return ExternalStateOptimizerState(
            name=name,
            step=0,
            first_moment=zeros,
            second_moment=zeros,
        )
    if name == "kfac":
        return ExternalStateOptimizerState(name=name, step=0)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def apply_external_state_optimizer_step(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    grads: Any,
    *,
    optimizer_state: ExternalStateOptimizerState | None,
    optimizer_name: str,
    learning_rate: float,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    lamb_eps: float = 1.0e-6,
    max_update_l2_norm: float | None = None,
) -> tuple[tuple[Any, ...], ExternalStateOptimizerState | None, Any]:
    """Apply SGD/Adam/LAMB updates to external state parameters."""

    name = _normalize_optimizer_name(optimizer_name)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if optimizer_state is None or optimizer_state.name != name:
        optimizer_state = init_external_state_optimizer(
            adapter,
            state_params,
            optimizer_name=name,
        )
    if name == "none":
        updates = adapter.modules.jax.tree_util.tree_map(
            lambda leaf: adapter.modules.jnp.zeros_like(leaf),
            state_params,
        )
        return state_params, optimizer_state, updates
    if name == "sgd":
        direction = _add_decoupled_weight_decay(adapter, grads, state_params, weight_decay)
        updates = adapter.modules.jax.tree_util.tree_map(
            lambda grad: -adapter.modules.jnp.asarray(learning_rate) * grad,
            direction,
        )
        updates = _clip_update_tree(adapter, updates, max_update_l2_norm)
        new_params = adapter.modules.jax.tree_util.tree_map(
            lambda param, update: param + update,
            state_params,
            updates,
        )
        return (
            new_params,
            ExternalStateOptimizerState(name=name, step=optimizer_state.step + 1),
            updates,
        )
    if name not in ("adam", "lamb"):
        return (
            state_params,
            init_external_state_optimizer(adapter, state_params, optimizer_name=name),
            None,
        )

    if optimizer_state.first_moment is None or optimizer_state.second_moment is None:
        optimizer_state = init_external_state_optimizer(
            adapter,
            state_params,
            optimizer_name=name,
        )
    next_step = optimizer_state.step + 1
    jnp = adapter.modules.jnp
    first = adapter.modules.jax.tree_util.tree_map(
        lambda m, g: adam_b1 * m + (1.0 - adam_b1) * g,
        optimizer_state.first_moment,
        grads,
    )
    second = adapter.modules.jax.tree_util.tree_map(
        lambda v, g: adam_b2 * v
        + (1.0 - adam_b2) * jnp.real(jnp.asarray(g) * jnp.conj(jnp.asarray(g))),
        optimizer_state.second_moment,
        grads,
    )
    b1_correction = 1.0 - adam_b1**next_step
    b2_correction = 1.0 - adam_b2**next_step

    def adam_direction(moment, variance, param):
        direction = (moment / b1_correction) / (
            jnp.sqrt(variance / b2_correction) + adam_eps
        )
        if weight_decay:
            direction = direction + weight_decay * param
        return direction

    direction = adapter.modules.jax.tree_util.tree_map(
        adam_direction,
        first,
        second,
        state_params,
    )
    if name == "lamb":
        direction = adapter.modules.jax.tree_util.tree_map(
            lambda param, update: _lamb_trust_scaled_direction(
                jnp,
                param,
                update,
                lamb_eps=lamb_eps,
            ),
            state_params,
            direction,
        )
    updates = adapter.modules.jax.tree_util.tree_map(
        lambda update: -jnp.asarray(learning_rate) * update,
        direction,
    )
    updates = _clip_update_tree(adapter, updates, max_update_l2_norm)
    new_params = adapter.modules.jax.tree_util.tree_map(
        lambda param, update: param + update,
        state_params,
        updates,
    )
    return (
        new_params,
        ExternalStateOptimizerState(
            name=name,
            step=next_step,
            first_moment=first,
            second_moment=second,
        ),
        updates,
    )


def apply_external_state_kfac_step(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    optimizer_state: ExternalStateOptimizerState | None,
    step: int,
    learning_rate: float,
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
    overlap_scale_state_energy: Any | None = None,
    overlap_scale_state_energy_std: Any | None = None,
    precomputed_terms: dict[str, Any] | None = None,
    max_update_l2_norm: float | None = None,
    kfac_damping: float | None = None,
    kfac_momentum: float | None = None,
    kfac_norm_constraint: float | None = None,
    kfac_l2_reg: float | None = None,
    kfac_cov_ema_decay: float | None = None,
    kfac_invert_every: int | None = None,
    kfac_register_only_generic: bool | None = None,
) -> tuple[tuple[Any, ...], ExternalStateOptimizerState, Any]:
    """Apply one direct KFAC update to external state parameters."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    jax = adapter.modules.jax
    jnp = adapter.modules.jnp
    kfac_jax, graph_patterns, pmap_axis_name = _load_kfac_modules()
    use_multi_device = _external_state_kfac_should_use_multi_device(adapter, samples)
    local_device_count = int(jax.local_device_count()) if use_multi_device else 1
    native_state_metadata = _external_state_kfac_native_state_metadata(
        use_multi_device=use_multi_device,
        device_count=local_device_count,
        walkers_per_state=samples.walkers,
        pmap_axis_name=pmap_axis_name,
    )
    damping = _kfac_config_value(adapter, "damping", 0.001, kfac_damping)
    momentum = _kfac_config_value(adapter, "momentum", 0.0, kfac_momentum)
    norm_constraint = _kfac_config_value(
        adapter,
        "norm_constraint",
        0.001,
        kfac_norm_constraint,
    )
    l2_reg = _kfac_config_value(adapter, "l2_reg", 0.0, kfac_l2_reg)
    cov_ema_decay = _kfac_config_value(
        adapter,
        "cov_ema_decay",
        0.95,
        kfac_cov_ema_decay,
    )
    invert_every = int(
        _kfac_config_value(adapter, "invert_every", 1, kfac_invert_every)
    )
    register_only_generic = bool(
        _kfac_config_value(
            adapter,
            "register_only_generic",
            False,
            kfac_register_only_generic,
        )
    )

    def objective(params, rng, batch):
        del rng
        sample_batch, batch_terms = _unpack_external_state_kfac_batch(batch)
        _register_external_state_kfac_predictive_distribution(
            adapter,
            params,
            sample_batch,
            kfac_jax,
        )
        value = ferminet_pbc_penalty_training_objective(
            adapter,
            params,
            sample_batch,
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
            precomputed_terms=batch_terms,
            overlap_scale_state_energy=overlap_scale_state_energy,
            overlap_scale_state_energy_std=overlap_scale_state_energy_std,
        )
        return value, {"loss": value}

    def estimator_value(params, rng, batch):
        del rng
        sample_batch, _ = _unpack_external_state_kfac_batch(batch)
        mean = _register_external_state_kfac_predictive_distribution(
            adapter,
            params,
            sample_batch,
            kfac_jax,
        )
        return jnp.mean(mean)

    if gradient_mode != "paper_tangent":
        raise ValueError("KFAC external-state updates require paper_tangent mode")
    if use_multi_device:
        kfac_batch = _pack_external_state_kfac_multi_device_batch(
            adapter,
            samples,
            precomputed_terms,
            kfac_jax=kfac_jax,
            pmap_axis_name=pmap_axis_name,
            device_count=local_device_count,
        )
    else:
        kfac_batch = _pack_external_state_kfac_batch(samples, precomputed_terms)
    optimizer_key = _external_state_kfac_optimizer_cache_key(
        adapter,
        state_params,
        samples,
        learning_rate=learning_rate,
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
        l2_reg=l2_reg,
        norm_constraint=norm_constraint,
        cov_ema_decay=cov_ema_decay,
        invert_every=invert_every,
        register_only_generic=register_only_generic,
        batch=kfac_batch,
        multi_device=use_multi_device,
        pmap_axis_name=pmap_axis_name,
    )
    optimizer = _EXTERNAL_STATE_KFAC_OPTIMIZER_CACHE.get(optimizer_key)
    if optimizer is None:
        val_and_grad = jax.value_and_grad(objective, argnums=0, has_aux=True)
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg=l2_reg,
            norm_constraint=norm_constraint,
            value_func_has_aux=True,
            value_func_has_rng=True,
            value_func_for_estimator=estimator_value,
            learning_rate_schedule=lambda _: jnp.asarray(learning_rate),
            curvature_ema=cov_ema_decay,
            inverse_update_period=invert_every,
            min_damping=_kfac_config_value(adapter, "min_damping", 1.0e-4, None),
            num_burnin_steps=0,
            register_only_generic=register_only_generic,
            estimation_mode="fisher_exact",
            multi_device=use_multi_device,
            pmap_axis_name=pmap_axis_name,
            auto_register_kwargs={"graph_patterns": graph_patterns},
            batch_size_extractor=_external_state_kfac_batch_size,
            include_norms_in_stats=True,
        )
        _EXTERNAL_STATE_KFAC_OPTIMIZER_CACHE[optimizer_key] = optimizer
    rng = jax.random.PRNGKey(9187 + int(step))
    if use_multi_device:
        rng = kfac_jax.utils.make_different_rng_key_on_all_devices(rng)
    native_state = (
        optimizer_state.native_state
        if (
            optimizer_state is not None
            and optimizer_state.name == "kfac"
            and _external_state_kfac_native_state_metadata_matches(
                optimizer_state.native_state_metadata,
                native_state_metadata,
            )
        )
        else None
    )
    opt_params = _copy_jax_tree(adapter, state_params)
    if use_multi_device:
        opt_params = _external_state_kfac_replicate_tree(
            kfac_jax,
            opt_params,
            pmap_axis_name=pmap_axis_name,
        )
    native_state_reused = native_state is not None
    if native_state is None:
        native_state = optimizer.init(opt_params, rng, kfac_batch)
    opt_native_state = _copy_jax_tree(adapter, native_state)
    step_momentum = jnp.asarray(momentum)
    step_damping = jnp.asarray(damping)
    if use_multi_device:
        step_momentum = _external_state_kfac_replicate_tree(
            kfac_jax,
            step_momentum,
            pmap_axis_name=pmap_axis_name,
        )
        step_damping = _external_state_kfac_replicate_tree(
            kfac_jax,
            step_damping,
            pmap_axis_name=pmap_axis_name,
        )
    new_params, native_state, stats = optimizer.step(
        params=opt_params,
        state=opt_native_state,
        rng=rng,
        batch=kfac_batch,
        momentum=step_momentum,
        damping=step_damping,
        global_step_int=int(optimizer_state.step)
        if native_state_reused and optimizer_state is not None
        else None,
    )
    if use_multi_device:
        new_params = _external_state_kfac_unreplicate_tree(adapter, new_params)
    updates = jax.tree_util.tree_map(
        lambda old, new: new - old,
        state_params,
        new_params,
    )
    clipped_updates = _clip_update_tree(adapter, updates, max_update_l2_norm)
    if max_update_l2_norm is not None:
        new_params = jax.tree_util.tree_map(
            lambda param, update: param + update,
            state_params,
            clipped_updates,
        )
        updates = clipped_updates
    next_step = _external_state_kfac_step_from_stats(adapter, stats)
    return (
        new_params,
        ExternalStateOptimizerState(
            name="kfac",
            step=next_step,
            native_state=native_state,
            native_state_metadata=native_state_metadata,
        ),
        updates,
    )


def _load_kfac_modules() -> tuple[Any, Any, str]:
    """Import kfac_jax and FermiNet graph patterns lazily."""

    import kfac_jax  # pylint: disable=import-outside-toplevel
    from ferminet import constants  # pylint: disable=import-outside-toplevel
    from ferminet import curvature_tags_and_blocks  # pylint: disable=import-outside-toplevel

    return kfac_jax, curvature_tags_and_blocks.GRAPH_PATTERNS, constants.PMAP_AXIS_NAME


def _copy_jax_tree(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Copy JAX array leaves before KFAC donates buffers internally."""

    return adapter.modules.jax.tree_util.tree_map(
        lambda leaf: leaf + adapter.modules.jnp.zeros_like(leaf)
        if hasattr(leaf, "shape")
        else leaf,
        tree,
    )


def _external_state_kfac_should_use_multi_device(
    adapter: FermiNetPBCExternalStateAdapter,
    samples: FermiNetPBCStateSamples,
) -> bool:
    """Return whether the current batch can be split over all local devices."""

    device_count = int(adapter.modules.jax.local_device_count())
    return (
        device_count > 1
        and samples.walkers >= device_count
        and samples.walkers % device_count == 0
    )


def _external_state_kfac_native_state_metadata(
    *,
    use_multi_device: bool,
    device_count: int,
    walkers_per_state: int,
    pmap_axis_name: str,
) -> dict[str, Any]:
    """Metadata that determines whether a cached KFAC state is reusable."""

    effective_device_count = int(device_count) if use_multi_device else 1
    return {
        "multi_device": bool(use_multi_device),
        "device_count": effective_device_count,
        "walkers_per_device": int(walkers_per_state) // effective_device_count,
        "pmap_axis_name": pmap_axis_name if use_multi_device else None,
    }


def _external_state_kfac_native_state_metadata_matches(
    current: Any,
    expected: dict[str, Any],
) -> bool:
    if current is None:
        return not bool(expected.get("multi_device", False))
    if not isinstance(current, dict):
        return False
    return current == expected


def _external_state_kfac_replicate_tree(
    kfac_jax: Any,
    tree: Any,
    *,
    pmap_axis_name: str,
) -> Any:
    return kfac_jax.utils.replicate_all_local_devices(
        tree,
        axis_name=pmap_axis_name,
    )


def _external_state_kfac_unreplicate_tree(
    adapter: FermiNetPBCExternalStateAdapter,
    tree: Any,
) -> Any:
    return adapter.modules.jax.tree_util.tree_map(
        lambda leaf: adapter.modules.jnp.asarray(leaf)[0]
        if hasattr(leaf, "shape") and len(adapter.modules.jnp.asarray(leaf).shape) > 0
        else leaf,
        tree,
    )


def _external_state_kfac_step_from_stats(
    adapter: FermiNetPBCExternalStateAdapter,
    stats: dict[str, Any],
) -> int:
    step = adapter.modules.jnp.asarray(stats["step"])
    if step.shape:
        step = adapter.modules.jnp.ravel(step)[0]
    return int(step)


def _register_external_state_kfac_predictive_distribution(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    kfac_jax: Any,
) -> Any:
    """Register external-state network outputs as KFAC predictive factors."""

    outputs = []
    for params in state_params:
        for sample_state in range(samples.nstates):
            positions, spins, atoms, charges = samples.for_sample_state(sample_state)
            _, logabs = adapter.batched_signed_network(
                params,
                positions,
                spins,
                atoms,
                charges,
            )
            outputs.append(adapter.modules.jnp.ravel(adapter.modules.jnp.real(logabs)))
    if outputs:
        mean = adapter.modules.jnp.concatenate(outputs, axis=0)[:, None]
        kfac_jax.register_normal_predictive_distribution(mean)
        return mean
    return adapter.modules.jnp.zeros((1, 1))


def _external_state_kfac_batch_size(samples: FermiNetPBCStateSamples) -> Any:
    if _is_packed_external_state_kfac_batch(samples):
        samples = samples[0]
    positions = samples[0] if isinstance(samples, tuple) else samples.positions
    if len(positions.shape) > 4:
        return positions.shape[0] * positions.shape[1] * positions.shape[2]
    return positions.shape[0] * positions.shape[1]


def _pack_external_state_kfac_batch(
    samples: FermiNetPBCStateSamples,
    precomputed_terms: dict[str, Any] | None,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any] | None]:
    return _pack_external_state_kfac_samples(samples), precomputed_terms


def _pack_external_state_kfac_multi_device_batch(
    adapter: FermiNetPBCExternalStateAdapter,
    samples: FermiNetPBCStateSamples,
    precomputed_terms: dict[str, Any] | None,
    *,
    kfac_jax: Any,
    pmap_axis_name: str,
    device_count: int,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any] | None]:
    sample_batch = tuple(
        _shard_external_state_kfac_walker_axis(adapter, leaf, device_count)
        for leaf in _pack_external_state_kfac_samples(samples)
    )
    if precomputed_terms is None:
        return sample_batch, None
    term_batch = _external_state_kfac_replicate_tree(
        kfac_jax,
        precomputed_terms,
        pmap_axis_name=pmap_axis_name,
    )
    if "local_energy" in precomputed_terms:
        term_batch = dict(term_batch)
        term_batch["local_energy"] = _shard_external_state_kfac_walker_axis(
            adapter,
            precomputed_terms["local_energy"],
            device_count,
        )
    return sample_batch, term_batch


def _shard_external_state_kfac_walker_axis(
    adapter: FermiNetPBCExternalStateAdapter,
    value: Any,
    device_count: int,
) -> Any:
    array = adapter.modules.jnp.asarray(value)
    if len(array.shape) < 2:
        raise ValueError("KFAC multi-device batch leaves need [states, walkers] axes")
    walkers = int(array.shape[1])
    if walkers % device_count != 0:
        raise ValueError(
            "walkers_per_state must be divisible by local_device_count for "
            "KFAC multi-device updates"
        )
    per_device = walkers // int(device_count)
    sharded = array.reshape(
        (array.shape[0], int(device_count), per_device, *array.shape[2:])
    )
    sharded = adapter.modules.jnp.swapaxes(sharded, 0, 1)
    local_devices = adapter.modules.jax.local_devices()[: int(device_count)]
    return adapter.modules.jax.device_put_sharded(
        [sharded[idx] for idx in range(int(device_count))],
        local_devices,
    )


def _unpack_external_state_kfac_batch(
    batch: Any,
) -> tuple[FermiNetPBCStateSamples, dict[str, Any] | None]:
    if _is_packed_external_state_kfac_batch(batch):
        return _unpack_external_state_kfac_samples(batch[0]), batch[1]
    return _unpack_external_state_kfac_samples(batch), None


def _is_packed_external_state_kfac_batch(batch: Any) -> bool:
    return (
        isinstance(batch, tuple)
        and len(batch) == 2
        and isinstance(batch[0], tuple)
        and len(batch[0]) == 4
    )


def _pack_external_state_kfac_samples(
    samples: FermiNetPBCStateSamples,
) -> tuple[Any, Any, Any, Any]:
    return samples.positions, samples.spins, samples.atoms, samples.charges


def _unpack_external_state_kfac_samples(
    batch: tuple[Any, Any, Any, Any],
) -> FermiNetPBCStateSamples:
    return FermiNetPBCStateSamples(
        positions=batch[0],
        spins=batch[1],
        atoms=batch[2],
        charges=batch[3],
    )


def _external_state_kfac_optimizer_cache_key(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    samples: FermiNetPBCStateSamples,
    *,
    learning_rate: float,
    penalty_alpha: float,
    local_energy: BatchedLocalEnergy | None,
    energy_weights: Any | None,
    collapse_threshold: float,
    clip_upper: bool,
    ratio_clip_width: float | None,
    ratio_exclude_width: float,
    max_logabs_ratio: float | None,
    local_energy_clip_width: float | None,
    local_energy_exclude_width: float,
    overlap_scale_by: str | None,
    min_gap_scale_factor: float,
    max_scale_factor: float,
    state_ordering: str,
    l2_reg: Any,
    norm_constraint: Any,
    cov_ema_decay: Any,
    invert_every: int,
    register_only_generic: bool,
    batch: Any,
    multi_device: bool,
    pmap_axis_name: str,
) -> tuple[Any, ...]:
    return (
        id(adapter),
        id(local_energy),
        id(energy_weights),
        _tree_shape_signature(adapter, state_params),
        _tree_shape_signature(adapter, samples),
        _tree_shape_signature(adapter, energy_weights),
        _tree_shape_signature(adapter, batch),
        float(learning_rate),
        float(penalty_alpha),
        float(collapse_threshold),
        bool(clip_upper),
        None if ratio_clip_width is None else float(ratio_clip_width),
        float(ratio_exclude_width),
        None if max_logabs_ratio is None else float(max_logabs_ratio),
        None if local_energy_clip_width is None else float(local_energy_clip_width),
        float(local_energy_exclude_width),
        None if overlap_scale_by is None else str(overlap_scale_by),
        float(min_gap_scale_factor),
        float(max_scale_factor),
        str(state_ordering),
        float(l2_reg),
        float(norm_constraint),
        float(cov_ema_decay),
        int(invert_every),
        bool(register_only_generic),
        bool(multi_device),
        str(pmap_axis_name) if multi_device else None,
    )


def _tree_shape_signature(
    adapter: FermiNetPBCExternalStateAdapter,
    tree: Any,
) -> Any:
    if tree is None:
        return None
    leaves, treedef = adapter.modules.jax.tree_util.tree_flatten(tree)
    return (
        str(treedef),
        tuple(
            (
                tuple(getattr(leaf, "shape", ())),
                str(getattr(leaf, "dtype", type(leaf).__name__)),
            )
            for leaf in leaves
        ),
    )


def _kfac_config_value(
    adapter: FermiNetPBCExternalStateAdapter,
    name: str,
    default: Any,
    override: Any | None,
) -> Any:
    if override is not None:
        return override
    try:
        return adapter.cfg.optim.kfac.get(name, default)
    except AttributeError:
        return default


def update_external_state_penalty_running_stats(
    adapter: FermiNetPBCExternalStateAdapter,
    running_stats: ExternalStatePenaltyRunningStats | None,
    terms: dict[str, Any],
    *,
    decay: float | None,
) -> ExternalStatePenaltyRunningStats | None:
    """Update EWMA energy/std statistics used by overlap scaling."""

    if decay is None:
        return running_stats
    if decay < 0.0 or decay >= 1.0:
        raise ValueError("overlap_ewma_decay must be in [0, 1)")
    jax = adapter.modules.jax
    energy = jax.lax.stop_gradient(terms["state_energy"])
    std = jax.lax.stop_gradient(terms["state_energy_std"])
    if running_stats is None:
        return ExternalStatePenaltyRunningStats(
            step=0,
            state_energy_ewma=energy,
            state_energy_std_ewma=std,
            decay=decay,
        )
    return ExternalStatePenaltyRunningStats(
        step=running_stats.step + 1,
        state_energy_ewma=decay * running_stats.state_energy_ewma
        + (1.0 - decay) * energy,
        state_energy_std_ewma=decay * running_stats.state_energy_std_ewma
        + (1.0 - decay) * std,
        decay=decay,
    )


def merge_external_state_params(
    adapter: FermiNetPBCExternalStateAdapter,
    state_params: tuple[Any, ...],
    merge_keys: tuple[str, ...] | list[str] | None,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Average selected parameter leaves across external states."""

    keys = _normalize_merge_keys(merge_keys)
    if not keys:
        return state_params, ()
    path_leaves, treedef = _tree_path_leaves(adapter, state_params[0])
    leaves_by_state = [adapter.modules.jax.tree_util.tree_leaves(params) for params in state_params]
    new_leaves_by_state = [list(leaves) for leaves in leaves_by_state]
    matched_paths = []
    for leaf_idx, (path, _) in enumerate(path_leaves):
        if not _path_matches(path, keys):
            continue
        values = [leaves[leaf_idx] for leaves in leaves_by_state]
        mean_value = sum(values) / len(values)
        for state_leaves in new_leaves_by_state:
            state_leaves[leaf_idx] = mean_value
        matched_paths.append(path)
    if not matched_paths:
        return state_params, ()
    return tuple(treedef.unflatten(leaves) for leaves in new_leaves_by_state), tuple(
        matched_paths
    )


def merge_external_optimizer_state(
    adapter: FermiNetPBCExternalStateAdapter,
    optimizer_state: ExternalStateOptimizerState | None,
    merge_keys: tuple[str, ...] | list[str] | None,
) -> ExternalStateOptimizerState | None:
    """Merge optimizer moments for parameters that are shared across states."""

    if optimizer_state is None:
        return None
    first = optimizer_state.first_moment
    second = optimizer_state.second_moment
    if first is not None:
        first, _ = merge_external_state_params(adapter, first, merge_keys)
    if second is not None:
        second, _ = merge_external_state_params(adapter, second, merge_keys)
    return ExternalStateOptimizerState(
        name=optimizer_state.name,
        step=optimizer_state.step,
        first_moment=first,
        second_moment=second,
        native_state=optimizer_state.native_state,
        native_state_metadata=optimizer_state.native_state_metadata,
    )


def _normalize_optimizer_name(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "": "sgd",
        "adamw": "adam",
        "optax_adam": "adam",
        "optax_lamb": "lamb",
    }
    return aliases.get(normalized, normalized)


def _add_decoupled_weight_decay(
    adapter: FermiNetPBCExternalStateAdapter,
    grads: Any,
    params: Any,
    weight_decay: float,
) -> Any:
    if not weight_decay:
        return grads
    return adapter.modules.jax.tree_util.tree_map(
        lambda grad, param: grad + weight_decay * param,
        grads,
        params,
    )


def _clip_update_tree(
    adapter: FermiNetPBCExternalStateAdapter,
    updates: Any,
    max_norm: float | None,
) -> Any:
    if max_norm is None:
        return updates
    update_norm = tree_l2_norm(adapter, updates)
    scale = adapter.modules.jnp.minimum(1.0, max_norm / (update_norm + 1.0e-30))
    return adapter.modules.jax.tree_util.tree_map(lambda leaf: leaf * scale, updates)


def _lamb_trust_scaled_direction(
    jnp: Any,
    params: Any,
    direction: Any,
    *,
    lamb_eps: float,
) -> Any:
    param_norm = jnp.linalg.norm(jnp.ravel(jnp.asarray(params)))
    direction_norm = jnp.linalg.norm(jnp.ravel(jnp.asarray(direction)))
    trust_ratio = jnp.where(
        (param_norm > 0.0) & (direction_norm > 0.0),
        param_norm / (direction_norm + lamb_eps),
        1.0,
    )
    return trust_ratio * direction


def _running_scale_inputs(
    running_stats: ExternalStatePenaltyRunningStats | None,
) -> tuple[Any | None, Any | None]:
    if running_stats is None:
        return None, None
    return running_stats.state_energy_ewma, running_stats.state_energy_std_ewma


def _select_optimizer_state(
    adapter: FermiNetPBCExternalStateAdapter,
    accepted: Any,
    old_state: ExternalStateOptimizerState | None,
    new_state: ExternalStateOptimizerState | None,
) -> ExternalStateOptimizerState | None:
    if new_state is None or old_state is None:
        return new_state if bool(accepted) else old_state
    if new_state.first_moment is None and new_state.second_moment is None:
        return new_state if bool(accepted) else old_state
    first = _select_tree(adapter, accepted, old_state.first_moment, new_state.first_moment)
    second = _select_tree(
        adapter,
        accepted,
        old_state.second_moment,
        new_state.second_moment,
    )
    return ExternalStateOptimizerState(
        name=new_state.name,
        step=new_state.step if bool(accepted) else old_state.step,
        first_moment=first,
        second_moment=second,
        native_state=new_state.native_state if bool(accepted) else old_state.native_state,
        native_state_metadata=new_state.native_state_metadata
        if bool(accepted)
        else old_state.native_state_metadata,
    )


def _select_tree(
    adapter: FermiNetPBCExternalStateAdapter,
    accepted: Any,
    old_tree: Any,
    new_tree: Any,
) -> Any:
    if old_tree is None or new_tree is None:
        return new_tree
    return adapter.modules.jax.tree_util.tree_map(
        lambda old, new: adapter.modules.jnp.where(accepted, new, old),
        old_tree,
        new_tree,
    )


def _normalize_merge_keys(
    merge_keys: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if merge_keys is None:
        return ()
    if isinstance(merge_keys, str):
        merge_keys = (merge_keys,)
    return tuple(str(key).strip() for key in merge_keys if str(key).strip())


def _tree_path_leaves(
    adapter: FermiNetPBCExternalStateAdapter,
    tree: Any,
) -> tuple[list[tuple[str, Any]], Any]:
    tree_util = adapter.modules.jax.tree_util
    if hasattr(tree_util, "tree_flatten_with_path"):
        path_leaves, treedef = tree_util.tree_flatten_with_path(tree)
        return [
            (_format_jax_tree_path(path), leaf)
            for path, leaf in path_leaves
        ], treedef
    leaves, treedef = tree_util.tree_flatten(tree)
    return [(str(idx), leaf) for idx, leaf in enumerate(leaves)], treedef


def _format_jax_tree_path(path: Any) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        else:
            text = str(entry)
            text = text.removeprefix("['").removesuffix("']")
            parts.append(text)
    return "/".join(parts)


def _path_matches(path: str, merge_keys: tuple[str, ...]) -> bool:
    if "*" in merge_keys:
        return True
    return any(key in path for key in merge_keys)


def tree_l2_norm(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Return the real-valued L2 norm of a JAX pytree."""

    jnp = adapter.modules.jnp
    leaves = adapter.modules.jax.tree_util.tree_leaves(tree)
    total = jnp.asarray(0.0)
    for leaf in leaves:
        array = jnp.asarray(leaf)
        total = total + jnp.sum(jnp.real(array * jnp.conj(array)))
    return jnp.sqrt(total)


def tree_all_finite(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Return whether every real and imaginary leaf entry is finite."""

    jnp = adapter.modules.jnp
    leaves = adapter.modules.jax.tree_util.tree_leaves(tree)
    finite = jnp.asarray(True)
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if jnp.iscomplexobj(array):
            leaf_finite = jnp.all(jnp.isfinite(jnp.real(array))) & jnp.all(
                jnp.isfinite(jnp.imag(array))
            )
        else:
            leaf_finite = jnp.all(jnp.isfinite(array))
        finite = finite & leaf_finite
    return finite


def scalar_all_finite(adapter: FermiNetPBCExternalStateAdapter, value: Any) -> Any:
    """Return whether a scalar or array value is finite."""

    jnp = adapter.modules.jnp
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        return jnp.all(jnp.isfinite(jnp.real(array))) & jnp.all(
            jnp.isfinite(jnp.imag(array))
        )
    return jnp.all(jnp.isfinite(array))


def sanitize_tree_nonfinite(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Replace non-finite gradient entries before forming guarded candidates."""

    jnp = adapter.modules.jnp

    def sanitize_leaf(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        if jnp.iscomplexobj(array):
            real = jnp.nan_to_num(jnp.real(array), nan=0.0, posinf=0.0, neginf=0.0)
            imag = jnp.nan_to_num(jnp.imag(array), nan=0.0, posinf=0.0, neginf=0.0)
            return real + 1j * imag
        return jnp.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    return adapter.modules.jax.tree_util.tree_map(sanitize_leaf, tree)


def clip_tree_by_global_norm(
    adapter: FermiNetPBCExternalStateAdapter,
    tree: Any,
    *,
    grad_norm: Any,
    max_norm: float,
) -> Any:
    """Clip a gradient pytree by global L2 norm."""

    jnp = adapter.modules.jnp
    scale = jnp.minimum(1.0, jnp.asarray(max_norm) / (grad_norm + 1.0e-30))
    return adapter.modules.jax.tree_util.tree_map(lambda leaf: leaf * scale, tree)


def penalty_terms_all_finite(
    adapter: FermiNetPBCExternalStateAdapter,
    terms: dict[str, Any],
) -> Any:
    """Return whether the main objective diagnostics are all finite."""

    jnp = adapter.modules.jnp
    finite = jnp.asarray(True)
    for key in (
        "local_energy",
        "state_energy",
        "state_energy_std",
        "overlap_matrix",
        "psi_ratio",
        "penalty_objective",
    ):
        array = jnp.asarray(terms[key])
        if jnp.iscomplexobj(array):
            key_finite = jnp.all(jnp.isfinite(jnp.real(array))) & jnp.all(
                jnp.isfinite(jnp.imag(array))
            )
        else:
            key_finite = jnp.all(jnp.isfinite(array))
        finite = finite & key_finite
    return finite


def block_tree_until_ready(adapter: FermiNetPBCExternalStateAdapter, tree: Any) -> Any:
    """Block on all JAX arrays in a pytree and return the same tree shape."""

    def block(value: Any) -> Any:
        if hasattr(value, "block_until_ready"):
            return value.block_until_ready()
        return value

    return adapter.modules.jax.tree_util.tree_map(block, tree)


__all__ = [
    "ExternalStateOptimizerState",
    "ExternalStatePenaltyRunningStats",
    "ExternalStatePenaltyStepDiagnostics",
    "ExternalStatePenaltyTrainingResult",
    "apply_external_state_kfac_step",
    "apply_external_state_optimizer_step",
    "block_tree_until_ready",
    "clip_tree_by_global_norm",
    "external_state_penalty_optimizer_step",
    "external_state_penalty_sgd_step",
    "init_external_state_optimizer",
    "merge_external_optimizer_state",
    "merge_external_state_params",
    "penalty_terms_all_finite",
    "run_external_state_penalty_sgd",
    "sanitize_tree_nonfinite",
    "scalar_all_finite",
    "tree_all_finite",
    "tree_l2_norm",
    "update_external_state_penalty_running_stats",
]
