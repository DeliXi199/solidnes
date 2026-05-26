"""Excited-state objective helpers for SolidNES."""

from solidnes.excited_states.ferminet_pbc_adapter import FermiNetJAXModules
from solidnes.excited_states.ferminet_pbc_adapter import FermiNetPBCExternalStateAdapter
from solidnes.excited_states.ferminet_pbc_adapter import apply_external_state_sgd_step
from solidnes.excited_states.ferminet_pbc_adapter import assert_pbc_external_state_config
from solidnes.excited_states.ferminet_pbc_adapter import build_external_state_adapter
from solidnes.excited_states.ferminet_pbc_adapter import configure_jax_platform
from solidnes.excited_states.ferminet_pbc_adapter import evaluate_ferminet_pbc_penalty_terms
from solidnes.excited_states.ferminet_pbc_adapter import ferminet_pbc_penalty_objective
from solidnes.excited_states.ferminet_pbc_adapter import (
    ferminet_pbc_penalty_training_objective,
)
from solidnes.excited_states.ferminet_pbc_adapter import init_external_state_params
from solidnes.excited_states.ferminet_pbc_adapter import load_ferminet_jax_modules
from solidnes.excited_states.ferminet_pbc_adapter import make_network_from_config
from solidnes.excited_states.ferminet_pbc_adapter import make_tiny_state_samples
from solidnes.excited_states.ferminet_pbc_adapter import (
    value_and_grad_ferminet_pbc_penalty_objective,
)
from solidnes.excited_states.ferminet_pbc_adapter import wrap_pbc_local_energy
from solidnes.excited_states.ferminet_pbc_adapter import wrap_signed_network
from solidnes.excited_states.ferminet_pbc_driver import ExternalStateDriverResult
from solidnes.excited_states.ferminet_pbc_driver import (
    ExternalStateDriverStepDiagnostics,
)
from solidnes.excited_states.ferminet_pbc_driver import (
    ExternalStateMetropolisDiagnostics,
)
from solidnes.excited_states.ferminet_pbc_driver import (
    load_external_state_driver_checkpoint,
)
from solidnes.excited_states.ferminet_pbc_driver import metropolis_update_state_samples
from solidnes.excited_states.ferminet_pbc_driver import (
    run_external_state_penalty_driver,
)
from solidnes.excited_states.ferminet_pbc_driver import (
    save_external_state_driver_checkpoint,
)
from solidnes.excited_states.ferminet_pbc_driver import wrap_flat_positions_in_lattice
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples
from solidnes.excited_states.ferminet_pbc_scaffold import StateEnergyEstimate
from solidnes.excited_states.ferminet_pbc_scaffold import StateWavefunctionMatrix
from solidnes.excited_states.ferminet_pbc_scaffold import broadcast_state_samples
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_overlap_diagnostics
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_penalty_scaffold_terms
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_energy_estimate
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_wavefunction_matrix
from solidnes.excited_states.ferminet_pbc_training import block_tree_until_ready
from solidnes.excited_states.ferminet_pbc_training import clip_tree_by_global_norm
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStateOptimizerState,
)
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStatePenaltyRunningStats,
)
from solidnes.excited_states.ferminet_pbc_training import (
    apply_external_state_kfac_step,
)
from solidnes.excited_states.ferminet_pbc_training import (
    apply_external_state_optimizer_step,
)
from solidnes.excited_states.ferminet_pbc_training import (
    external_state_penalty_sgd_step,
)
from solidnes.excited_states.ferminet_pbc_training import (
    external_state_penalty_optimizer_step,
)
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStatePenaltyStepDiagnostics,
)
from solidnes.excited_states.ferminet_pbc_training import (
    ExternalStatePenaltyTrainingResult,
)
from solidnes.excited_states.ferminet_pbc_training import (
    init_external_state_optimizer,
)
from solidnes.excited_states.ferminet_pbc_training import (
    merge_external_optimizer_state,
)
from solidnes.excited_states.ferminet_pbc_training import (
    merge_external_state_params,
)
from solidnes.excited_states.ferminet_pbc_training import (
    run_external_state_penalty_sgd,
)
from solidnes.excited_states.ferminet_pbc_training import (
    update_external_state_penalty_running_stats,
)
from solidnes.excited_states.ferminet_pbc_training import penalty_terms_all_finite
from solidnes.excited_states.ferminet_pbc_training import tree_all_finite
from solidnes.excited_states.ferminet_pbc_training import tree_l2_norm
from solidnes.excited_states.overlap import clip_psi_ratios_by_median
from solidnes.excited_states.overlap import estimate_overlap_from_ratios
from solidnes.excited_states.overlap import offdiag_squared_overlap
from solidnes.excited_states.overlap import overlap_diagnostics
from solidnes.excited_states.overlap import overlap_penalty_loss
from solidnes.excited_states.overlap import scaled_offdiag_squared_overlap
from solidnes.excited_states.overlap import symmetrize_overlap_with_clipped_geometric_mean
from solidnes.excited_states.ferminet_pbc_scaffold import wavefunction_ratios_from_matrix
from solidnes.excited_states.penalty import energy_gap_scale
from solidnes.excited_states.penalty import energy_std_scale
from solidnes.excited_states.penalty import max_gap_std_scale
from solidnes.excited_states.penalty import overlap_gradient_scale
from solidnes.excited_states.penalty import penalty_vmc_loss
from solidnes.excited_states.penalty import penalty_vmc_terms
from solidnes.excited_states.penalty import weighted_state_energy

__all__ = [
    "FermiNetJAXModules",
    "FermiNetPBCExternalStateAdapter",
    "FermiNetPBCStateSamples",
    "ExternalStatePenaltyStepDiagnostics",
    "ExternalStatePenaltyTrainingResult",
    "ExternalStateDriverResult",
    "ExternalStateDriverStepDiagnostics",
    "ExternalStateMetropolisDiagnostics",
    "ExternalStateOptimizerState",
    "ExternalStatePenaltyRunningStats",
    "StateEnergyEstimate",
    "StateWavefunctionMatrix",
    "apply_external_state_kfac_step",
    "apply_external_state_optimizer_step",
    "apply_external_state_sgd_step",
    "assert_pbc_external_state_config",
    "block_tree_until_ready",
    "broadcast_state_samples",
    "build_external_state_adapter",
    "clip_psi_ratios_by_median",
    "clip_tree_by_global_norm",
    "configure_jax_platform",
    "energy_gap_scale",
    "energy_std_scale",
    "external_state_penalty_optimizer_step",
    "external_state_penalty_sgd_step",
    "evaluate_overlap_diagnostics",
    "evaluate_penalty_scaffold_terms",
    "evaluate_state_energy_estimate",
    "evaluate_state_wavefunction_matrix",
    "evaluate_ferminet_pbc_penalty_terms",
    "estimate_overlap_from_ratios",
    "ferminet_pbc_penalty_objective",
    "ferminet_pbc_penalty_training_objective",
    "init_external_state_params",
    "init_external_state_optimizer",
    "load_external_state_driver_checkpoint",
    "load_ferminet_jax_modules",
    "make_network_from_config",
    "make_tiny_state_samples",
    "max_gap_std_scale",
    "merge_external_optimizer_state",
    "merge_external_state_params",
    "metropolis_update_state_samples",
    "offdiag_squared_overlap",
    "overlap_diagnostics",
    "overlap_gradient_scale",
    "overlap_penalty_loss",
    "penalty_terms_all_finite",
    "penalty_vmc_loss",
    "penalty_vmc_terms",
    "run_external_state_penalty_sgd",
    "run_external_state_penalty_driver",
    "save_external_state_driver_checkpoint",
    "scaled_offdiag_squared_overlap",
    "symmetrize_overlap_with_clipped_geometric_mean",
    "tree_all_finite",
    "tree_l2_norm",
    "update_external_state_penalty_running_stats",
    "value_and_grad_ferminet_pbc_penalty_objective",
    "wrap_pbc_local_energy",
    "wrap_signed_network",
    "wrap_flat_positions_in_lattice",
    "wavefunction_ratios_from_matrix",
    "weighted_state_energy",
]
