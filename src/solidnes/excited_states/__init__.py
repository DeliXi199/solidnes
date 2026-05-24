"""Excited-state objective helpers for SolidNES."""

from solidnes.excited_states.overlap import estimate_overlap_from_ratios
from solidnes.excited_states.overlap import offdiag_squared_overlap
from solidnes.excited_states.overlap import overlap_diagnostics
from solidnes.excited_states.overlap import overlap_penalty_loss
from solidnes.excited_states.overlap import symmetrize_overlap_with_clipped_geometric_mean
from solidnes.excited_states.penalty import energy_gap_scale
from solidnes.excited_states.penalty import energy_std_scale
from solidnes.excited_states.penalty import max_gap_std_scale
from solidnes.excited_states.penalty import penalty_vmc_loss
from solidnes.excited_states.penalty import penalty_vmc_terms
from solidnes.excited_states.penalty import weighted_state_energy

__all__ = [
    "energy_gap_scale",
    "energy_std_scale",
    "estimate_overlap_from_ratios",
    "max_gap_std_scale",
    "offdiag_squared_overlap",
    "overlap_diagnostics",
    "overlap_penalty_loss",
    "penalty_vmc_loss",
    "penalty_vmc_terms",
    "symmetrize_overlap_with_clipped_geometric_mean",
    "weighted_state_energy",
]
