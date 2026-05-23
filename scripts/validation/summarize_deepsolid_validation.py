#!/usr/bin/env python
"""Summarize DeepSolid training stats for a short validation run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solidnes.backends.deepsolid_adapter import build_deepsolid_adapter


def _default_validation_dir(bundle) -> Path:
    output = bundle.experiment.get("output", {})
    validation_dir = output.get("validation_dir")
    if validation_dir:
        return PROJECT_ROOT / validation_dir
    return Path(bundle.cfg.log.save_path).parent / "validation"


def _read_stats(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "energy": float(row["energy"]),
                    "variance": float(row["variance"]),
                    "pmove": float(row["pmove"]),
                    "imaginary": float(row["imaginary"]),
                }
            )
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _block_means(values: list[float], block_count: int) -> list[float]:
    if block_count <= 0 or len(values) < block_count:
        return []
    block_size = len(values) // block_count
    blocks = []
    for start in range(0, block_size * block_count, block_size):
        blocks.append(_mean(values[start : start + block_size]))
    return blocks


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient"
    delta = values[-1] - values[0]
    if delta < 0.0:
        return "down"
    if delta > 0.0:
        return "up"
    return "flat"


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _load_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.12g}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    hf = summary.get("hf_reference") or {}
    basis = summary.get("basis", "configured")
    lines = [
        "# Carbon Diamond Validation Summary",
        "",
        f"Experiment: `{summary['experiment_name']}`",
        f"Created: `{summary['created_at']}`",
        "",
        "## Training Stats",
        "",
        f"- Rows: {summary['rows']}",
        f"- First step: {summary['first_step']}",
        f"- Last step: {summary['last_step']}",
        f"- First energy: {summary['first_energy_hartree']:.12g} Ha",
        f"- Last energy: {summary['last_energy_hartree']:.12g} Ha",
        f"- Minimum energy: {summary['min_energy_hartree']:.12g} Ha",
        f"- Energy delta: {summary['energy_delta_hartree']:.12g} Ha",
        f"- Energy trend: {summary['energy_trend']}",
        f"- First variance: {summary['first_variance']:.12g}",
        f"- Last variance: {summary['last_variance']:.12g}",
        f"- Variance trend: {summary['variance_trend']}",
        f"- Mean pmove: {summary['pmove_mean']:.12g}",
        f"- Pmove range: [{summary['pmove_min']:.12g}, {summary['pmove_max']:.12g}]",
        f"- Finite checks: {summary['finite_checks_passed']}",
        "",
        "## Tail / Block Estimate",
        "",
        f"- Tail rows: {summary['tail_rows']}",
        f"- Tail start step: {summary['tail_start_step']}",
        f"- Tail energy mean: {summary['tail_energy_mean_hartree']:.12g} Ha",
        f"- Tail energy stderr: {_format_optional(summary['tail_energy_stderr_hartree'])} Ha",
        f"- Block count: {summary['block_count_used']}",
        f"- Block energy stderr: {_format_optional(summary['block_energy_stderr_hartree'])} Ha",
        f"- Tail variance mean: {summary['tail_variance_mean']:.12g}",
        f"- Tail pmove mean: {summary['tail_pmove_mean']:.12g}",
        "",
    ]
    if hf:
        lines.extend(
            [
                "## HF Reference",
                "",
                f"- Reference: {hf.get('reference')}",
                f"- Converged: {hf.get('converged')}",
                f"- HF total energy: {hf.get('e_tot_hartree'):.12g} Ha",
                f"- VMC last minus HF: {summary['last_minus_hf_hartree']:.12g} Ha",
                f"- VMC min minus HF: {summary['min_minus_hf_hartree']:.12g} Ha",
                f"- VMC tail mean minus HF: {summary['tail_mean_minus_hf_hartree']:.12g} Ha",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            f"This is a validation harness for the configured carbon `{basis}` setup.",
            "It checks finite execution, trends, and HF comparison for the same",
            "setup; it is not by itself a production carbon benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment/diamond_c_deepsolid_validation_short.yaml",
        help="Path to a SolidNES experiment YAML, relative to project root.",
    )
    parser.add_argument("--stats", default=None, help="Path to train_stats.csv.")
    parser.add_argument("--reference", default=None, help="Path to HF reference JSON.")
    parser.add_argument("--output-json", default=None, help="Output summary JSON path.")
    parser.add_argument("--output-md", default=None, help="Output Markdown report path.")
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.5,
        help="Fraction of final rows used for tail averages and error estimates.",
    )
    parser.add_argument(
        "--block-count",
        type=int,
        default=5,
        help="Number of blocks for a simple block standard error estimate.",
    )
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1]")

    bundle = build_deepsolid_adapter(PROJECT_ROOT / args.experiment)
    validation_dir = _default_validation_dir(bundle)
    stats_path = Path(args.stats).resolve() if args.stats else Path(bundle.cfg.log.save_path) / "train_stats.csv"
    reference_path = (
        Path(args.reference).resolve()
        if args.reference
        else validation_dir / "pyscf_pbc_hf_reference.json"
    )
    output_json = Path(args.output_json).resolve() if args.output_json else validation_dir / "training_summary.json"
    output_md = Path(args.output_md).resolve() if args.output_md else validation_dir / "training_summary.md"

    rows = _read_stats(stats_path)
    energies = [row["energy"] for row in rows]
    variances = [row["variance"] for row in rows]
    pmoves = [row["pmove"] for row in rows]
    imaginaries = [row["imaginary"] for row in rows]
    reference = _load_reference(reference_path)
    tail_count = max(1, int(math.ceil(len(rows) * args.tail_fraction)))
    tail_rows = rows[-tail_count:]
    tail_energies = [row["energy"] for row in tail_rows]
    tail_variances = [row["variance"] for row in tail_rows]
    tail_pmoves = [row["pmove"] for row in tail_rows]
    energy_block_means = _block_means(tail_energies, args.block_count)

    summary: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": bundle.experiment["experiment_name"],
        "basis": bundle.summary.basis,
        "stats_path": str(stats_path),
        "reference_path": str(reference_path) if reference else None,
        "rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "first_energy_hartree": energies[0],
        "last_energy_hartree": energies[-1],
        "min_energy_hartree": min(energies),
        "energy_delta_hartree": energies[-1] - energies[0],
        "energy_trend": _trend(energies),
        "first_variance": variances[0],
        "last_variance": variances[-1],
        "variance_delta": variances[-1] - variances[0],
        "variance_trend": _trend(variances),
        "pmove_mean": _mean(pmoves),
        "pmove_min": min(pmoves),
        "pmove_max": max(pmoves),
        "imaginary_abs_max": max(abs(value) for value in imaginaries),
        "finite_checks_passed": _finite(energies + variances + pmoves + imaginaries),
        "tail_fraction": args.tail_fraction,
        "tail_rows": len(tail_rows),
        "tail_start_step": tail_rows[0]["step"],
        "tail_energy_mean_hartree": _mean(tail_energies),
        "tail_energy_stderr_hartree": _std_error(tail_energies),
        "tail_variance_mean": _mean(tail_variances),
        "tail_pmove_mean": _mean(tail_pmoves),
        "block_count_requested": args.block_count,
        "block_count_used": len(energy_block_means),
        "block_energy_means_hartree": energy_block_means,
        "block_energy_stderr_hartree": _std_error(energy_block_means),
    }
    if reference:
        hf_energy = float(reference["e_tot_hartree"])
        summary["hf_reference"] = reference
        summary["last_minus_hf_hartree"] = energies[-1] - hf_energy
        summary["min_minus_hf_hartree"] = min(energies) - hf_energy
        summary["tail_mean_minus_hf_hartree"] = summary["tail_energy_mean_hartree"] - hf_energy

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _write_markdown(output_md, summary)

    print(f"experiment: {summary['experiment_name']}")
    print(f"rows: {summary['rows']}")
    print(f"energy_first: {summary['first_energy_hartree']:.12g}")
    print(f"energy_last: {summary['last_energy_hartree']:.12g}")
    print(f"energy_delta: {summary['energy_delta_hartree']:.12g}")
    print(f"variance_delta: {summary['variance_delta']:.12g}")
    print(f"pmove_mean: {summary['pmove_mean']:.12g}")
    print(f"tail_energy_mean: {summary['tail_energy_mean_hartree']:.12g}")
    print(f"tail_energy_stderr: {_format_optional(summary['tail_energy_stderr_hartree'])}")
    print(f"block_energy_stderr: {_format_optional(summary['block_energy_stderr_hartree'])}")
    if reference:
        print(f"hf_energy: {reference['e_tot_hartree']:.12g}")
        print(f"last_minus_hf: {summary['last_minus_hf_hartree']:.12g}")
        print(f"tail_mean_minus_hf: {summary['tail_mean_minus_hf_hartree']:.12g}")
    print(f"summary_json: {output_json}")
    print(f"summary_md: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
