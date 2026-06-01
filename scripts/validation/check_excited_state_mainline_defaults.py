#!/usr/bin/env python
"""Check that the source-code excited-state mainline resolves to no-merge 0096."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (SRC_ROOT, FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from solidnes.backends.ferminet_adapter import build_ferminet_adapter
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_ATTENTION
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_INDEPENDENT_STATE_PARAMS
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_METHOD
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_MERGE_KEYS
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_MODEL_CONFIG
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_NETWORK
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_OBJECTIVE
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_REFERENCE_EXPERIMENT
from solidnes.excited_state_mainline import MAINLINE_EXCITED_STATE_TRAIN_CONFIG
from solidnes.excited_state_mainline import resolve_mainline_attention


AUTO_SMOKE_EXPERIMENT = (
    "configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_auto_smoke.yaml"
)
FERMINET_CONTROL_EXPERIMENT = (
    "configs/experiment/"
    "diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_"
    "ferminet_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mainline-experiment",
        default=MAINLINE_EXCITED_STATE_REFERENCE_EXPERIMENT,
        help="Reference experiment expected to be on the 0096 mainline.",
    )
    parser.add_argument(
        "--auto-experiment",
        default=AUTO_SMOKE_EXPERIMENT,
        help="Auto-attention experiment expected to resolve to the mainline.",
    )
    parser.add_argument(
        "--control-experiment",
        default=FERMINET_CONTROL_EXPERIMENT,
        help="FermiNet-attention control expected to remain non-mainline.",
    )
    args = parser.parse_args()

    _assert_path_exists(MAINLINE_EXCITED_STATE_MODEL_CONFIG)
    _assert_path_exists(MAINLINE_EXCITED_STATE_TRAIN_CONFIG)

    mainline = build_ferminet_adapter(PROJECT_ROOT / args.mainline_experiment)
    _assert_mainline_summary(mainline.summary)
    _assert_deepqmc_alignment_defaults(mainline.summary)
    _assert_equal(
        _relative_path(mainline.paths.model),
        MAINLINE_EXCITED_STATE_MODEL_CONFIG,
        "mainline model config",
    )
    _assert_equal(
        _relative_path(mainline.paths.train),
        MAINLINE_EXCITED_STATE_TRAIN_CONFIG,
        "mainline train config",
    )

    auto = build_ferminet_adapter(PROJECT_ROOT / args.auto_experiment)
    _assert_mainline_summary(auto.summary)
    _assert_equal(
        resolve_mainline_attention(auto.summary.psiformer_attention_implementation),
        MAINLINE_EXCITED_STATE_ATTENTION,
        "auto attention resolution",
    )

    control = build_ferminet_adapter(PROJECT_ROOT / args.control_experiment)
    _assert_equal(
        control.summary.excited_state_route_role,
        "control",
        "FermiNet attention control role",
    )
    _assert_equal(
        control.summary.excited_state_route_is_mainline,
        False,
        "FermiNet attention control mainline flag",
    )

    print("excited_state_mainline_defaults: ok")
    print(f"method: {MAINLINE_EXCITED_STATE_METHOD}")
    print(f"reference_experiment: {args.mainline_experiment}")
    print(f"auto_experiment: {args.auto_experiment}")
    print(f"control_experiment: {args.control_experiment}")
    return 0


def _assert_mainline_summary(summary) -> None:
    _assert_equal(summary.objective, MAINLINE_EXCITED_STATE_OBJECTIVE, "objective")
    _assert_equal(summary.network_type, MAINLINE_EXCITED_STATE_NETWORK, "network_type")
    _assert_equal(summary.states, 2, "states")
    _assert_equal(
        summary.independent_state_params,
        MAINLINE_EXCITED_STATE_INDEPENDENT_STATE_PARAMS,
        "independent per-state params",
    )
    _assert_equal(summary.excited_state_route, MAINLINE_EXCITED_STATE_METHOD, "route")
    _assert_equal(summary.excited_state_route_role, "mainline", "route role")
    _assert_equal(summary.excited_state_route_is_mainline, True, "mainline flag")
    _assert_equal(
        resolve_mainline_attention(summary.psiformer_attention_implementation),
        MAINLINE_EXCITED_STATE_ATTENTION,
        "attention resolution",
    )


def _assert_deepqmc_alignment_defaults(summary) -> None:
    _assert_equal(summary.overlap_weights, (0.5, 0.5), "overlap weights")
    _assert_equal(summary.independent_state_params, True, "independent per-state params")
    _assert_equal(
        summary.independent_state_merge_keys,
        MAINLINE_EXCITED_STATE_MERGE_KEYS,
        "merge keys",
    )
    _assert_equal(summary.diagonal_mcmc_trace, True, "diagonal MCMC trace")
    _assert_equal(summary.diagonal_local_energy, True, "diagonal local energy")
    _assert_equal(summary.diagonal_overlap_jvp, True, "diagonal overlap JVP")
    _assert_equal(summary.profile_loss_components, False, "loss-component profiling")
    _assert_equal(summary.overlap_sort_states_by, None, "state ordering")
    _assert_equal(
        summary.kfac_norm_constraint_scale_by_states,
        False,
        "KFAC norm state scaling",
    )
    _assert_equal(summary.kfac_norm_constraint, 0.001, "KFAC norm constraint")


def _assert_path_exists(path: str) -> None:
    if not (PROJECT_ROOT / path).exists():
        raise AssertionError(f"missing configured mainline path: {path}")


def _assert_equal(actual, expected, name: str) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
