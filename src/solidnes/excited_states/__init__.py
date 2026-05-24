"""Excited-state objective helpers for SolidNES."""

from solidnes.excited_states.ferminet_pbc_adapter import FermiNetJAXModules
from solidnes.excited_states.ferminet_pbc_adapter import FermiNetPBCExternalStateAdapter
from solidnes.excited_states.ferminet_pbc_adapter import assert_pbc_external_state_config
from solidnes.excited_states.ferminet_pbc_adapter import build_external_state_adapter
from solidnes.excited_states.ferminet_pbc_adapter import configure_jax_platform
from solidnes.excited_states.ferminet_pbc_adapter import init_external_state_params
from solidnes.excited_states.ferminet_pbc_adapter import load_ferminet_jax_modules
from solidnes.excited_states.ferminet_pbc_adapter import make_network_from_config
from solidnes.excited_states.ferminet_pbc_adapter import make_tiny_state_samples
from solidnes.excited_states.ferminet_pbc_adapter import wrap_pbc_local_energy
from solidnes.excited_states.ferminet_pbc_adapter import wrap_signed_network
from solidnes.excited_states.ferminet_pbc_scaffold import FermiNetPBCStateSamples
from solidnes.excited_states.ferminet_pbc_scaffold import StateEnergyEstimate
from solidnes.excited_states.ferminet_pbc_scaffold import StateWavefunctionMatrix
from solidnes.excited_states.ferminet_pbc_scaffold import broadcast_state_samples
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_overlap_diagnostics
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_penalty_scaffold_terms
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_energy_estimate
from solidnes.excited_states.ferminet_pbc_scaffold import evaluate_state_wavefunction_matrix
from solidnes.excited_states.overlap import estimate_overlap_from_ratios
from solidnes.excited_states.overlap import offdiag_squared_overlap
from solidnes.excited_states.overlap import overlap_diagnostics
from solidnes.excited_states.overlap import overlap_penalty_loss
from solidnes.excited_states.overlap import symmetrize_overlap_with_clipped_geometric_mean
from solidnes.excited_states.ferminet_pbc_scaffold import wavefunction_ratios_from_matrix
from solidnes.excited_states.penalty import energy_gap_scale
from solidnes.excited_states.penalty import energy_std_scale
from solidnes.excited_states.penalty import max_gap_std_scale
from solidnes.excited_states.penalty import penalty_vmc_loss
from solidnes.excited_states.penalty import penalty_vmc_terms
from solidnes.excited_states.penalty import weighted_state_energy

__all__ = [
    "FermiNetJAXModules",
    "FermiNetPBCExternalStateAdapter",
    "FermiNetPBCStateSamples",
    "StateEnergyEstimate",
    "StateWavefunctionMatrix",
    "assert_pbc_external_state_config",
    "broadcast_state_samples",
    "build_external_state_adapter",
    "configure_jax_platform",
    "energy_gap_scale",
    "energy_std_scale",
    "evaluate_overlap_diagnostics",
    "evaluate_penalty_scaffold_terms",
    "evaluate_state_energy_estimate",
    "evaluate_state_wavefunction_matrix",
    "estimate_overlap_from_ratios",
    "init_external_state_params",
    "load_ferminet_jax_modules",
    "make_network_from_config",
    "make_tiny_state_samples",
    "max_gap_std_scale",
    "offdiag_squared_overlap",
    "overlap_diagnostics",
    "overlap_penalty_loss",
    "penalty_vmc_loss",
    "penalty_vmc_terms",
    "symmetrize_overlap_with_clipped_geometric_mean",
    "wrap_pbc_local_energy",
    "wrap_signed_network",
    "wavefunction_ratios_from_matrix",
    "weighted_state_energy",
]
