"""Source-of-truth defaults for SolidNES excited-state production runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MAINLINE_EXCITED_STATE_METHOD = "0096_psiformer_vmc_overlap_fused_qkv"
MAINLINE_EXCITED_STATE_REFERENCE_TASK = (
    "tasks/psiformer/0103_attention_merge_keys_4gpu_10000"
)
MAINLINE_EXCITED_STATE_REFERENCE_EXPERIMENT = (
    "configs/experiment/"
    "diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_"
    "merge_none_batch4096_iter10000.yaml"
)
MAINLINE_EXCITED_STATE_MODEL_CONFIG = (
    "configs/model/psiformer_pbc_paper_attention_fused_qkv_x64_jaxattn.yaml"
)
MAINLINE_EXCITED_STATE_TRAIN_CONFIG = (
    "configs/train/"
    "excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_"
    "batch4096_iter10000.yaml"
)
MAINLINE_EXCITED_STATE_OBJECTIVE = "vmc_overlap"
MAINLINE_EXCITED_STATE_NETWORK = "psiformer"
MAINLINE_EXCITED_STATE_ATTENTION = "fused_qkv"
MAINLINE_EXCITED_STATE_ATTENTION_KERNEL_GPU = "jax"
MAINLINE_EXCITED_STATE_METHOD_PROFILE = "szabo_noe_2024_penalty"
MAINLINE_EXCITED_STATE_MIN_STATES = 2
MAINLINE_EXCITED_STATE_INDEPENDENT_STATE_PARAMS = True
MAINLINE_EXCITED_STATE_MERGE_KEYS: tuple[str, ...] = ()

# Optimizer schedule selected from the 0113/0114 fresh-start eta/tau sweep.
# This is the default learning-rate schedule for future excited-state runs
# unless a task explicitly declares a new sweep or ablation.
DEFAULT_EXCITED_STATE_LEARNING_RATE = 0.02
DEFAULT_EXCITED_STATE_LEARNING_RATE_DELAY = 10000.0
DEFAULT_EXCITED_STATE_LEARNING_RATE_DECAY = 1.0
DEFAULT_EXCITED_STATE_OPTIMIZER_EVIDENCE = (
    "tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_"
    "low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/"
    "0113_0114_fixed_tau_eta_comparison_last10000_comparison.md"
)
DEFAULT_EXCITED_STATE_REFERENCE_TRAIN_CONFIG = (
    "configs/train/"
    "excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_damp1e3_"
    "default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml"
)
DEFAULT_EXCITED_STATE_REFERENCE_EXPERIMENT = (
    "configs/experiment/"
    "diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_"
    "damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml"
)


@dataclass(frozen=True)
class ExcitedStateMainlineSelection:
    """Classification of a FermiNet/PsiFormer excited-state configuration."""

    method: str | None
    role: str | None
    is_mainline: bool
    reason: str
    resolved_attention: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "role": self.role,
            "is_mainline": self.is_mainline,
            "reason": self.reason,
            "resolved_attention": self.resolved_attention,
        }


def resolve_mainline_attention(implementation: str | None) -> str | None:
    """Resolve a PsiFormer attention selector the same way production runs do."""

    if implementation is None:
        return None
    value = str(implementation).lower()
    if value == "auto":
        return MAINLINE_EXCITED_STATE_ATTENTION
    if value in {"ferminet", "upstream", "default"}:
        return "ferminet"
    if value == MAINLINE_EXCITED_STATE_ATTENTION:
        return value
    return value


def classify_excited_state_mainline(
    *,
    objective: str | None,
    states: int | None,
    network_type: str | None,
    attention_implementation: str | None = None,
    attention_kernel_gpu: str | None = None,
    method_profile: str | None = None,
    independent_state_params: bool | None = None,
    independent_state_merge_keys: tuple[str, ...] | list[str] | None = None,
) -> ExcitedStateMainlineSelection:
    """Classify whether a config is on the no-merge excited-state mainline."""

    objective_value = _lower_or_none(objective)
    network_value = _lower_or_none(network_type)
    state_count = int(states or 0)
    resolved_attention = resolve_mainline_attention(attention_implementation)
    kernel_value = _lower_or_none(attention_kernel_gpu)
    profile_value = _lower_or_none(method_profile)
    independent_value = bool(independent_state_params)
    merge_keys = tuple(str(key) for key in (independent_state_merge_keys or ()))

    if objective_value != MAINLINE_EXCITED_STATE_OBJECTIVE:
        return ExcitedStateMainlineSelection(
            method=None,
            role=None,
            is_mainline=False,
            reason="not a native vmc_overlap excited-state objective",
            resolved_attention=resolved_attention,
        )
    if state_count < MAINLINE_EXCITED_STATE_MIN_STATES:
        return ExcitedStateMainlineSelection(
            method=None,
            role=None,
            is_mainline=False,
            reason="vmc_overlap requires at least two states for the mainline",
            resolved_attention=resolved_attention,
        )
    if network_value != MAINLINE_EXCITED_STATE_NETWORK:
        return ExcitedStateMainlineSelection(
            method="ferminet_native_vmc_overlap",
            role="legacy_or_control",
            is_mainline=False,
            reason="native vmc_overlap is active, but the network is not PsiFormer",
            resolved_attention=resolved_attention,
        )
    if independent_value != MAINLINE_EXCITED_STATE_INDEPENDENT_STATE_PARAMS:
        return ExcitedStateMainlineSelection(
            method="psiformer_vmc_overlap_shared_state_params",
            role="legacy_or_control",
            is_mainline=False,
            reason="mainline excited-state route requires independent per-state parameters",
            resolved_attention=resolved_attention,
        )
    if merge_keys != MAINLINE_EXCITED_STATE_MERGE_KEYS:
        return ExcitedStateMainlineSelection(
            method=MAINLINE_EXCITED_STATE_METHOD,
            role="merge_key_variant",
            is_mainline=False,
            reason="mainline architecture with optional merge_keys parameter sharing",
            resolved_attention=resolved_attention,
        )
    if resolved_attention != MAINLINE_EXCITED_STATE_ATTENTION:
        return ExcitedStateMainlineSelection(
            method="psiformer_vmc_overlap_attention_control",
            role="control",
            is_mainline=False,
            reason="PsiFormer vmc_overlap control run with non-mainline attention",
            resolved_attention=resolved_attention,
        )
    if (
        kernel_value
        and kernel_value != MAINLINE_EXCITED_STATE_ATTENTION_KERNEL_GPU
    ):
        return ExcitedStateMainlineSelection(
            method=MAINLINE_EXCITED_STATE_METHOD,
            role="mainline_with_kernel_override",
            is_mainline=True,
            reason="mainline method with an explicit attention-kernel override",
            resolved_attention=resolved_attention,
        )
    if (
        profile_value
        and profile_value != MAINLINE_EXCITED_STATE_METHOD_PROFILE
    ):
        return ExcitedStateMainlineSelection(
            method=MAINLINE_EXCITED_STATE_METHOD,
            role="mainline_with_profile_override",
            is_mainline=True,
            reason="mainline architecture with an explicit method-profile override",
            resolved_attention=resolved_attention,
        )
    return ExcitedStateMainlineSelection(
        method=MAINLINE_EXCITED_STATE_METHOD,
        role="mainline",
        is_mainline=True,
        reason="0096 PsiFormer vmc_overlap fused-QKV no-merge excited-state route",
        resolved_attention=resolved_attention,
    )


def _lower_or_none(value: str | None) -> str | None:
    return None if value is None else str(value).lower()


__all__ = [
    "ExcitedStateMainlineSelection",
    "DEFAULT_EXCITED_STATE_LEARNING_RATE",
    "DEFAULT_EXCITED_STATE_LEARNING_RATE_DECAY",
    "DEFAULT_EXCITED_STATE_LEARNING_RATE_DELAY",
    "DEFAULT_EXCITED_STATE_OPTIMIZER_EVIDENCE",
    "DEFAULT_EXCITED_STATE_REFERENCE_EXPERIMENT",
    "DEFAULT_EXCITED_STATE_REFERENCE_TRAIN_CONFIG",
    "MAINLINE_EXCITED_STATE_ATTENTION",
    "MAINLINE_EXCITED_STATE_ATTENTION_KERNEL_GPU",
    "MAINLINE_EXCITED_STATE_INDEPENDENT_STATE_PARAMS",
    "MAINLINE_EXCITED_STATE_METHOD",
    "MAINLINE_EXCITED_STATE_METHOD_PROFILE",
    "MAINLINE_EXCITED_STATE_MERGE_KEYS",
    "MAINLINE_EXCITED_STATE_MIN_STATES",
    "MAINLINE_EXCITED_STATE_MODEL_CONFIG",
    "MAINLINE_EXCITED_STATE_NETWORK",
    "MAINLINE_EXCITED_STATE_OBJECTIVE",
    "MAINLINE_EXCITED_STATE_REFERENCE_EXPERIMENT",
    "MAINLINE_EXCITED_STATE_REFERENCE_TASK",
    "MAINLINE_EXCITED_STATE_TRAIN_CONFIG",
    "classify_excited_state_mainline",
    "resolve_mainline_attention",
]
