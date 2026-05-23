#!/usr/bin/env python
"""Summarize FermiNet benchmark runs from stats and SLURM logs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (SRC_ROOT, FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from solidnes.backends.ferminet_adapter import build_ferminet_adapter


START_RE = re.compile(r"FermiNet GPU smoke starts at (.+)")
END_RE = re.compile(r"FermiNet GPU smoke ends at (.+)")
WARNING_PATTERNS = {
    "folx_tile_not_in_registry": "tile not in registry",
    "traceback": "Traceback",
}


def main() -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="SolidNES FermiNet experiment YAML.")
    parser.add_argument("--stats", default=None, help="Path to train_stats.csv.")
    parser.add_argument("--log", default=None, help="Path to SLURM stdout log.")
    parser.add_argument("--err", default=None, help="Path to SLURM stderr log.")
    parser.add_argument("--plan-json", default=None, help="Path to SLURM plan JSON.")
    parser.add_argument("--job-id", default=None, help="SLURM job id for metadata.")
    parser.add_argument("--output-json", default=None, help="Output summary JSON path.")
    parser.add_argument("--output-md", default=None, help="Output Markdown report path.")
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--block-count", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1]")

    bundle = build_ferminet_adapter(PROJECT_ROOT / args.experiment)
    checkpoint_dir = Path(bundle.cfg.log.save_path)
    validation_dir = checkpoint_dir.parent / "validation"
    stats_path = Path(args.stats).resolve() if args.stats else checkpoint_dir / "train_stats.csv"
    log_path = Path(args.log).resolve() if args.log else None
    err_path = Path(args.err).resolve() if args.err else None
    plan_path = Path(args.plan_json).resolve() if args.plan_json else None
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else validation_dir / "benchmark_summary.json"
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else validation_dir / "benchmark_summary.md"
    )

    rows = _read_stats(stats_path)
    log_text = _read_optional_text(log_path)
    err_text = _read_optional_text(err_path)
    plan = _read_optional_json(plan_path)
    runtime = _runtime_summary(log_text)
    energies = [row["energy"] for row in rows]
    ewmeans = [row["ewmean"] for row in rows]
    ewvars = [row["ewvar"] for row in rows]
    pmoves = [row["pmove"] for row in rows]
    tail_count = max(1, int(math.ceil(len(rows) * args.tail_fraction)))
    tail_rows = rows[-tail_count:]
    tail_energies = [row["energy"] for row in tail_rows]
    tail_ewmeans = [row["ewmean"] for row in tail_rows]
    tail_pmoves = [row["pmove"] for row in tail_rows]
    block_means = _block_means(tail_energies, args.block_count)
    warning_counts = _warning_counts(log_text + "\n" + err_text)
    slurm = _slurm_summary(plan)

    summary: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": bundle.experiment["experiment_name"],
        "job_id": args.job_id,
        "stats_path": str(stats_path),
        "log_path": str(log_path) if log_path else None,
        "err_path": str(err_path) if err_path else None,
        "plan_json_path": str(plan_path) if plan_path else None,
        "runtime": runtime,
        "slurm": slurm,
        "config": {
            "backend": bundle.summary.backend,
            "network_type": bundle.summary.network_type,
            "determinants": bundle.summary.determinants,
            "hidden_dims": str(bundle.summary.hidden_dims),
            "batch_size": bundle.summary.batch_size,
            "optimizer": bundle.summary.optimizer,
            "iterations": bundle.summary.iterations,
            "laplacian": bundle.summary.laplacian,
            "forward_laplacian_enabled": bundle.summary.forward_laplacian_enabled,
            "mcmc_burn_in": bundle.summary.mcmc_burn_in,
            "mcmc_steps_per_iteration": bundle.summary.mcmc_steps_per_iteration,
            "target_jax_version": bundle.summary.target_jax_version,
            "precision_profile": bundle.summary.precision_profile,
            "x64_enabled": bundle.summary.x64_enabled,
        },
        "stats": {
            "rows": len(rows),
            "first_step": rows[0]["step"],
            "last_step": rows[-1]["step"],
            "first_energy_hartree": energies[0],
            "last_energy_hartree": energies[-1],
            "min_energy_hartree": min(energies),
            "min_energy_step": rows[energies.index(min(energies))]["step"],
            "energy_delta_hartree": energies[-1] - energies[0],
            "first10_energy_mean_hartree": _prefix_mean(energies, 10),
            "last10_energy_mean_hartree": _suffix_mean(energies, 10),
            "first50_energy_mean_hartree": _prefix_mean(energies, 50),
            "last50_energy_mean_hartree": _suffix_mean(energies, 50),
            "first_ewmean_hartree": ewmeans[0],
            "last_ewmean_hartree": ewmeans[-1],
            "last_ewvar": ewvars[-1],
            "pmove_mean": _mean(pmoves),
            "pmove_min": min(pmoves),
            "pmove_max": max(pmoves),
            "finite_checks_passed": _finite(energies + ewmeans + ewvars + pmoves),
            "tail_fraction": args.tail_fraction,
            "tail_rows": len(tail_rows),
            "tail_start_step": tail_rows[0]["step"],
            "tail_energy_mean_hartree": _mean(tail_energies),
            "tail_energy_stderr_hartree": _std_error(tail_energies),
            "tail_ewmean_mean_hartree": _mean(tail_ewmeans),
            "tail_pmove_mean": _mean(tail_pmoves),
            "block_count_requested": args.block_count,
            "block_count_used": len(block_means),
            "block_energy_means_hartree": block_means,
            "block_energy_stderr_hartree": _std_error(block_means),
        },
        "diagnostics": warning_counts,
    }
    if runtime["elapsed_seconds"] is not None:
        summary["runtime"]["seconds_per_stats_row"] = runtime["elapsed_seconds"] / len(rows)
        update_steps = max(1, rows[-1]["step"] - rows[0]["step"] + 1)
        summary["runtime"]["seconds_per_optimization_step"] = (
            runtime["elapsed_seconds"] / update_steps
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, summary)
    _print_summary(summary, output_json, output_md)
    return 0


def _read_stats(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "step": int(row["step"]),
                    "energy": float(row["energy"]),
                    "ewmean": float(row["ewmean"]),
                    "ewvar": float(row["ewvar"]),
                    "pmove": float(row["pmove"]),
                }
            )
    if not rows:
        raise ValueError(f"No stats rows found in {path}")
    return rows


def _read_optional_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _prefix_mean(values: list[float], count: int) -> float | None:
    return _mean(values[:count]) if len(values) >= count else None


def _suffix_mean(values: list[float], count: int) -> float | None:
    return _mean(values[-count:]) if len(values) >= count else None


def _block_means(values: list[float], block_count: int) -> list[float]:
    if block_count <= 0 or len(values) < block_count:
        return []
    block_size = len(values) // block_count
    return [
        _mean(values[start : start + block_size])
        for start in range(0, block_size * block_count, block_size)
    ]


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _runtime_summary(log_text: str) -> dict[str, Any]:
    start = _find_datetime(log_text, START_RE)
    end = _find_datetime(log_text, END_RE)
    elapsed_seconds = (end - start).total_seconds() if start and end else None
    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "elapsed_seconds": elapsed_seconds,
    }


def _find_datetime(text: str, pattern: re.Pattern[str]) -> dt.datetime | None:
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return _parse_slurm_date(match.group(1).strip())
    return None


def _parse_slurm_date(value: str) -> dt.datetime | None:
    pieces = value.split()
    if len(pieces) == 6:
        value = " ".join(pieces[:4] + pieces[5:])
    try:
        return dt.datetime.strptime(value, "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None


def _warning_counts(text: str) -> dict[str, int]:
    return {name: text.count(pattern) for name, pattern in WARNING_PATTERNS.items()}


def _slurm_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {}
    request = plan.get("sbatch", {}).get("request", {})
    selected_node = plan.get("selection", {}).get("selected_node", {})
    config = plan.get("config", {})
    return {
        "ready": plan.get("ready"),
        "reason": plan.get("reason"),
        "partition": request.get("partition"),
        "gres": request.get("gres"),
        "cpus": request.get("cpus"),
        "selected_node": selected_node.get("name"),
        "gpu_model": selected_node.get("gpu_model_key"),
        "blocked_partitions": config.get("blocked_partitions", []),
    }


def _format_optional(value: Any, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    stats = summary["stats"]
    runtime = summary["runtime"]
    diagnostics = summary["diagnostics"]
    config = summary["config"]
    slurm = summary["slurm"]
    lines = [
        "# FermiNet Benchmark Summary",
        "",
        f"Experiment: `{summary['experiment_name']}`",
        f"Created: `{summary['created_at']}`",
        "",
        "## Runtime",
        "",
        f"- Job ID: `{summary.get('job_id') or 'n/a'}`",
        f"- Partition: `{slurm.get('partition', 'n/a')}`",
        f"- Node: `{slurm.get('selected_node', 'n/a')}`",
        f"- GPU model: `{slurm.get('gpu_model', 'n/a')}`",
        f"- GRES: `{slurm.get('gres', 'n/a')}`",
        f"- Elapsed seconds: {_format_optional(runtime.get('elapsed_seconds'))}",
        f"- Seconds per stats row: {_format_optional(runtime.get('seconds_per_stats_row'))}",
        f"- Seconds per optimization step: {_format_optional(runtime.get('seconds_per_optimization_step'))}",
        "",
        "## Config",
        "",
        f"- Network: `{config['network_type']}`",
        f"- Determinants: `{config['determinants']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Optimizer: `{config['optimizer']}`",
        f"- Iterations: `{config['iterations']}`",
        f"- Laplacian: `{config['laplacian']}`",
        f"- Forward Laplacian enabled: `{config['forward_laplacian_enabled']}`",
        f"- Precision profile: `{config['precision_profile']}`",
        f"- X64 enabled: `{config['x64_enabled']}`",
        "",
        "## Stats",
        "",
        f"- Rows: {stats['rows']}",
        f"- First/last step: {stats['first_step']} / {stats['last_step']}",
        f"- First energy: {stats['first_energy_hartree']:.6f} Ha",
        f"- Last energy: {stats['last_energy_hartree']:.6f} Ha",
        f"- Minimum energy: {stats['min_energy_hartree']:.6f} Ha at step {stats['min_energy_step']}",
        f"- Energy delta: {stats['energy_delta_hartree']:.6f} Ha",
        f"- First-10 mean: {_format_optional(stats['first10_energy_mean_hartree'])} Ha",
        f"- Last-10 mean: {_format_optional(stats['last10_energy_mean_hartree'])} Ha",
        f"- First-50 mean: {_format_optional(stats['first50_energy_mean_hartree'])} Ha",
        f"- Last-50 mean: {_format_optional(stats['last50_energy_mean_hartree'])} Ha",
        f"- Mean pmove: {stats['pmove_mean']:.6f}",
        f"- Pmove range: [{stats['pmove_min']:.6f}, {stats['pmove_max']:.6f}]",
        f"- Last ewvar: {stats['last_ewvar']:.6f}",
        f"- Finite checks: `{stats['finite_checks_passed']}`",
        "",
        "## Tail Estimate",
        "",
        f"- Tail rows: {stats['tail_rows']}",
        f"- Tail start step: {stats['tail_start_step']}",
        f"- Tail energy mean: {stats['tail_energy_mean_hartree']:.6f} Ha",
        f"- Tail energy stderr: {_format_optional(stats['tail_energy_stderr_hartree'])} Ha",
        f"- Block energy stderr: {_format_optional(stats['block_energy_stderr_hartree'])} Ha",
        f"- Tail pmove mean: {stats['tail_pmove_mean']:.6f}",
        "",
        "## Diagnostics",
        "",
        f"- FOLX tile warnings: {diagnostics['folx_tile_not_in_registry']}",
        f"- Traceback count: {diagnostics['traceback']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    stats = summary["stats"]
    runtime = summary["runtime"]
    diagnostics = summary["diagnostics"]
    print(f"experiment: {summary['experiment_name']}")
    print(f"rows: {stats['rows']}")
    print(f"energy_first: {stats['first_energy_hartree']:.12g}")
    print(f"energy_last: {stats['last_energy_hartree']:.12g}")
    print(f"energy_min: {stats['min_energy_hartree']:.12g}")
    print(f"pmove_mean: {stats['pmove_mean']:.12g}")
    print(f"elapsed_seconds: {_format_optional(runtime.get('elapsed_seconds'))}")
    print(f"seconds_per_step: {_format_optional(runtime.get('seconds_per_optimization_step'))}")
    print(f"folx_tile_warnings: {diagnostics['folx_tile_not_in_registry']}")
    print(f"tracebacks: {diagnostics['traceback']}")
    print(f"summary_json: {output_json}")
    print(f"summary_md: {output_md}")


if __name__ == "__main__":
    raise SystemExit(main())
