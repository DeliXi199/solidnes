#!/usr/bin/env python
"""Summarize a FermiNet PBC excited-state driver run summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", help="Driver run summary JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults beside the summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = _resolve_project_path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = (
        _resolve_project_path(args.output_dir)
        if args.output_dir
        else summary_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row(item) for item in summary.get("history", [])]
    csv_path = output_dir / "ferminet_pbc_driver_trajectory.csv"
    md_path = output_dir / "ferminet_pbc_driver_trajectory_analysis.md"
    _write_csv(csv_path, rows)
    analysis = _analysis(summary, rows)
    md_path.write_text(_format_markdown(summary, analysis, rows), encoding="utf-8")

    print("ferminet_pbc_driver_trajectory_summary: ok")
    print(f"summary_json: {_display_path(summary_path)}")
    print(f"trajectory_csv: {_display_path(csv_path)}")
    print(f"analysis_markdown: {_display_path(md_path)}")
    print(f"iterations: {analysis['iterations']}")
    print(f"objective_first: {analysis['objective_first']:.12g}")
    print(f"objective_final: {analysis['objective_final']:.12g}")
    print(f"objective_min: {analysis['objective_min']:.12g}")
    print(f"acceptance_mean: {analysis['acceptance_mean']:.12g}")
    print(f"grad_norm_max: {analysis['grad_norm_max']:.12g}")
    print(f"final_overlap_offdiag: {analysis['final_overlap_offdiag']:.12g}")
    return 0


def _row(item: dict[str, Any]) -> dict[str, float | int]:
    state_energy = item.get("state_energy", [])
    shared_paths = item.get("shared_param_paths", [])
    return {
        "iteration": int(item["iteration"]),
        "sampler_steps": int(item["sampler_steps"]),
        "sampler_acceptance": float(item["sampler_acceptance"]),
        "penalty_objective": float(item["penalty_objective"]),
        "gradient_objective": float(item["gradient_objective"]),
        "state0_energy": float(state_energy[0]),
        "state1_energy": float(state_energy[1]),
        "state0_energy_std": _safe_list_float(item.get("state_energy_std"), 0),
        "state1_energy_std": _safe_list_float(item.get("state_energy_std"), 1),
        "offdiag_squared_overlap": float(item["offdiag_squared_overlap"]),
        "grad_l2_norm": float(item["grad_l2_norm"]),
        "param_delta_l2_norm": float(item["param_delta_l2_norm"]),
        "optimizer_update_l2_norm": float(
            item.get("optimizer_update_l2_norm", item["param_delta_l2_norm"])
        ),
        "share_projection_l2_norm": float(item.get("share_projection_l2_norm", 0.0)),
        "optimizer_step": int(item.get("optimizer_step") or item["iteration"] + 1),
        "candidate_check_performed": int(bool(item.get("candidate_check_performed", True))),
        "shared_param_path_count": len(shared_paths),
        "update_accepted": int(bool(item["update_accepted"])),
    }


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analysis(
    summary: dict[str, Any],
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    objectives = [float(row["penalty_objective"]) for row in rows]
    acceptances = [float(row["sampler_acceptance"]) for row in rows]
    grad_norms = [float(row["grad_l2_norm"]) for row in rows]
    update_norms = [float(row["param_delta_l2_norm"]) for row in rows]
    optimizer_update_norms = [float(row["optimizer_update_l2_norm"]) for row in rows]
    share_projection_norms = [float(row["share_projection_l2_norm"]) for row in rows]
    candidate_checks = [int(row["candidate_check_performed"]) for row in rows]
    shared_counts = [int(row["shared_param_path_count"]) for row in rows]
    final_overlap = summary.get("final_overlap_matrix", [[1.0, 0.0], [0.0, 1.0]])
    return {
        "iterations": len(rows),
        "objective_first": objectives[0] if objectives else float("nan"),
        "objective_final": float(summary["final_penalty_objective"]),
        "objective_min": min(objectives) if objectives else float("nan"),
        "objective_max": max(objectives) if objectives else float("nan"),
        "acceptance_mean": sum(acceptances) / len(acceptances) if acceptances else float("nan"),
        "acceptance_min": min(acceptances) if acceptances else float("nan"),
        "acceptance_max": max(acceptances) if acceptances else float("nan"),
        "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
        "grad_norm_final": grad_norms[-1] if grad_norms else float("nan"),
        "update_norm_max": max(update_norms) if update_norms else float("nan"),
        "update_norm_final": update_norms[-1] if update_norms else float("nan"),
        "optimizer_update_norm_max": max(optimizer_update_norms)
        if optimizer_update_norms
        else float("nan"),
        "share_projection_norm_max": max(share_projection_norms)
        if share_projection_norms
        else float("nan"),
        "candidate_check_count": sum(candidate_checks),
        "shared_param_path_count_max": max(shared_counts) if shared_counts else 0,
        "final_overlap_offdiag": float(final_overlap[0][1]),
    }


def _format_markdown(
    summary: dict[str, Any],
    analysis: dict[str, float | int],
    rows: list[dict[str, float | int]],
) -> str:
    lines = [
        "# FermiNet PBC Driver Trajectory Analysis",
        "",
        "```text",
        f"experiment: {summary['experiment']}",
        f"jax_platform: {summary['jax_platform']}",
        f"walkers_per_state: {summary['walkers_per_state']}",
        f"completed_iterations: {summary['completed_iterations']}",
        f"optimizer: {summary.get('optimizer', 'sgd')}",
        f"overlap_ewma_decay: {summary.get('overlap_ewma_decay')}",
        f"param_share_keys: {summary.get('param_share_keys', [])}",
        f"candidate_check_period: {summary.get('candidate_check_period', 1)}",
        f"local_energy_source: {summary['local_energy_source']}",
        f"elapsed_seconds: {summary['elapsed_seconds']:.6f}",
        f"objective_first: {analysis['objective_first']:.12g}",
        f"objective_final: {analysis['objective_final']:.12g}",
        f"objective_min: {analysis['objective_min']:.12g}",
        f"acceptance_mean: {analysis['acceptance_mean']:.12g}",
        f"acceptance_range: [{analysis['acceptance_min']:.12g}, {analysis['acceptance_max']:.12g}]",
        f"grad_norm_max: {analysis['grad_norm_max']:.12g}",
        f"grad_norm_final: {analysis['grad_norm_final']:.12g}",
        f"update_norm_max: {analysis['update_norm_max']:.12g}",
        f"update_norm_final: {analysis['update_norm_final']:.12g}",
        f"optimizer_update_norm_max: {analysis['optimizer_update_norm_max']:.12g}",
        f"share_projection_norm_max: {analysis['share_projection_norm_max']:.12g}",
        f"candidate_check_count: {analysis['candidate_check_count']}",
        f"shared_param_path_count_max: {analysis['shared_param_path_count_max']}",
        f"final_overlap_offdiag: {analysis['final_overlap_offdiag']:.12g}",
        "```",
        "",
        "| iter | objective | acc | E0 | E1 | offdiag^2 | grad_norm | opt_update | share_proj | cand | shared |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['iteration']} | "
            f"{row['penalty_objective']:.6g} | "
            f"{row['sampler_acceptance']:.6g} | "
            f"{row['state0_energy']:.6g} | "
            f"{row['state1_energy']:.6g} | "
            f"{row['offdiag_squared_overlap']:.6g} | "
            f"{row['grad_l2_norm']:.6g} | "
            f"{row['optimizer_update_l2_norm']:.6g} | "
            f"{row['share_projection_l2_norm']:.6g} | "
            f"{row['candidate_check_performed']} | "
            f"{row['shared_param_path_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _safe_list_float(value: Any, idx: int) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value[idx])
    except (TypeError, IndexError, ValueError):
        return float("nan")


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
