#!/usr/bin/env python
"""Summarize FermiNet PBC-HF pretraining timing runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

START_RE = re.compile(r"FermiNet GPU smoke starts at (.+)")
END_RE = re.compile(r"FermiNet GPU smoke ends at (.+)")
PRETRAIN_RE = re.compile(
    r"PBC pretrain iter\s+(?P<step>\d+):\s+"
    r"loss=(?P<loss>\S+)\s+"
    r"pmove=(?P<pmove>\S+)\s+"
    r"step_seconds=(?P<step_seconds>\S+)\s+"
    r"target_eval_seconds=(?P<target_eval_seconds>\S+)\s+"
    r"target_transfer_seconds=(?P<target_transfer_seconds>\S+)\s+"
    r"jax_update_seconds=(?P<jax_update_seconds>\S+)"
)
WARNING_PATTERNS = {
    "folx_tile_not_in_registry": "tile not in registry",
    "traceback": "Traceback",
}
TIMING_KEYS = (
    "step_seconds",
    "target_eval_seconds",
    "target_transfer_seconds",
    "jax_update_seconds",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="SolidNES FermiNet experiment YAML.")
    parser.add_argument(
        "--pretrain-stats", default=None, help="Path to pretrain_stats.csv."
    )
    parser.add_argument("--log", default=None, help="Path to SLURM stdout log.")
    parser.add_argument("--err", default=None, help="Path to SLURM stderr log.")
    parser.add_argument("--plan-json", default=None, help="Path to SLURM plan JSON.")
    parser.add_argument("--job-id", default=None, help="SLURM job id for metadata.")
    parser.add_argument("--output-json", default=None, help="Output summary JSON path.")
    parser.add_argument("--output-md", default=None, help="Output Markdown report path.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Initial rows to exclude from steady-state timing summaries.",
    )
    args = parser.parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")

    experiment_path = _resolve_path(args.experiment)
    experiment = _read_yaml(experiment_path)
    output = experiment.get("output", {})
    checkpoint_dir = _resolve_path(output.get("checkpoint_dir", "."))
    validation_dir = _resolve_path(output.get("validation_dir", checkpoint_dir.parent))
    stats_path = (
        _resolve_path(args.pretrain_stats)
        if args.pretrain_stats
        else checkpoint_dir / "pretrain_stats.csv"
    )
    log_path = _resolve_path(args.log) if args.log else None
    err_path = _resolve_path(args.err) if args.err else None
    plan_path = _resolve_path(args.plan_json) if args.plan_json else None
    output_json = (
        _resolve_path(args.output_json)
        if args.output_json
        else validation_dir / "pretrain_benchmark_summary.json"
    )
    output_md = (
        _resolve_path(args.output_md)
        if args.output_md
        else validation_dir / "pretrain_benchmark_summary.md"
    )

    log_text = _read_optional_text(log_path)
    err_text = _read_optional_text(err_path)
    rows, row_source = _load_rows(stats_path, err_text)
    plan = _read_optional_json(plan_path)
    runtime = _runtime_summary(log_text)
    diagnostics = _warning_counts(log_text + "\n" + err_text)
    steady_rows = rows[args.warmup_steps :] if len(rows) > args.warmup_steps else rows

    summary: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": experiment.get("experiment_name"),
        "job_id": args.job_id,
        "experiment_path": str(experiment_path),
        "pretrain_stats_path": str(stats_path),
        "row_source": row_source,
        "log_path": str(log_path) if log_path else None,
        "err_path": str(err_path) if err_path else None,
        "plan_json_path": str(plan_path) if plan_path else None,
        "runtime": runtime,
        "slurm": _slurm_summary(plan),
        "config": _config_summary(experiment),
        "stats": _stats_summary(rows),
        "timing": _timing_summary(rows, steady_rows, args.warmup_steps),
        "diagnostics": diagnostics,
    }
    if runtime["elapsed_seconds"] is not None:
        summary["runtime"]["seconds_per_pretrain_row"] = (
            runtime["elapsed_seconds"] / len(rows)
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, summary)
    _print_summary(summary, output_json, output_md)
    return 0


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_rows(stats_path: Path, err_text: str) -> tuple[list[dict[str, float]], str]:
    if stats_path.exists():
        return _read_pretrain_stats(stats_path), "pretrain_stats_csv"
    rows = _parse_pretrain_log(err_text)
    if rows:
        return rows, "stderr_log"
    raise ValueError(
        f"No pretrain rows found in {stats_path} or supplied stderr log"
    )


def _read_pretrain_stats(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(_coerce_row(row))
    if not rows:
        raise ValueError(f"No pretrain stats rows found in {path}")
    return rows


def _parse_pretrain_log(text: str) -> list[dict[str, float]]:
    rows = []
    for line in text.splitlines():
        match = PRETRAIN_RE.search(line)
        if match:
            rows.append(_coerce_row(match.groupdict()))
    return rows


def _coerce_row(row: dict[str, str]) -> dict[str, float]:
    return {
        "step": int(row["step"]),
        "loss": float(row["loss"]),
        "pmove": float(row["pmove"]),
        "step_seconds": float(row["step_seconds"]),
        "target_eval_seconds": float(row["target_eval_seconds"]),
        "target_transfer_seconds": float(row["target_transfer_seconds"]),
        "jax_update_seconds": float(row["jax_update_seconds"]),
    }


def _read_optional_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stats_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    losses = [row["loss"] for row in rows]
    pmoves = [row["pmove"] for row in rows]
    min_loss = min(losses)
    min_loss_step = rows[losses.index(min_loss)]["step"]
    return {
        "rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
        "loss_delta": losses[-1] - losses[0],
        "loss_drop_fraction": (losses[0] - losses[-1]) / losses[0]
        if losses[0] != 0.0
        else None,
        "first10_loss_mean": _prefix_mean(losses, 10),
        "last10_loss_mean": _suffix_mean(losses, 10),
        "pmove_mean": _mean(pmoves),
        "pmove_min": min(pmoves),
        "pmove_max": max(pmoves),
        "finite_checks_passed": _finite(losses + pmoves),
    }


def _timing_summary(
    rows: list[dict[str, float]],
    steady_rows: list[dict[str, float]],
    warmup_steps: int,
) -> dict[str, Any]:
    timing: dict[str, Any] = {
        "warmup_steps_excluded": warmup_steps,
        "steady_rows": len(steady_rows),
        "compile_row": rows[0],
    }
    for prefix, source_rows in (("all", rows), ("steady", steady_rows)):
        for key in TIMING_KEYS:
            values = [row[key] for row in source_rows]
            timing[f"{prefix}_mean_{key}"] = _mean(values)
            timing[f"{prefix}_median_{key}"] = _quantile(values, 0.5)
            timing[f"{prefix}_p90_{key}"] = _quantile(values, 0.9)
            timing[f"{prefix}_max_{key}"] = max(values)
    steady_step = timing["steady_mean_step_seconds"]
    if steady_step:
        timing["steady_target_eval_fraction"] = (
            timing["steady_mean_target_eval_seconds"] / steady_step
        )
        timing["steady_target_transfer_fraction"] = (
            timing["steady_mean_target_transfer_seconds"] / steady_step
        )
        timing["steady_jax_update_fraction"] = (
            timing["steady_mean_jax_update_seconds"] / steady_step
        )
    return timing


def _config_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    notes = experiment.get("notes", {})
    pretraining = notes.get("pretraining", {})
    runtime = experiment.get("runtime", {})
    output = experiment.get("output", {})
    return {
        "phase": experiment.get("phase"),
        "precision_profile": runtime.get("precision_profile"),
        "x64_enabled": runtime.get("x64_enabled"),
        "require_gpu": runtime.get("require_gpu"),
        "checkpoint_dir": output.get("checkpoint_dir"),
        "validation_dir": output.get("validation_dir"),
        "pretrain_method": pretraining.get("method"),
        "pretrain_target_backend": pretraining.get("target_backend"),
        "pretrain_basis": pretraining.get("basis"),
        "pretrain_image_cutoff": pretraining.get("image_cutoff"),
        "pretrain_batch_size": pretraining.get("batch_size"),
        "pretrain_iterations": pretraining.get("iterations"),
        "pretrain_log_every": pretraining.get("log_every"),
        "pretrain_target_chunk_size": pretraining.get("target_chunk_size"),
    }


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


def _warning_counts(text: str) -> dict[str, int]:
    return {name: text.count(pattern) for name, pattern in WARNING_PATTERNS.items()}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _prefix_mean(values: list[float], count: int) -> float | None:
    return _mean(values[:count]) if len(values) >= count else None


def _suffix_mean(values: list[float], count: int) -> float | None:
    return _mean(values[-count:]) if len(values) >= count else None


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _format_optional(value: Any, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    stats = summary["stats"]
    timing = summary["timing"]
    runtime = summary["runtime"]
    diagnostics = summary["diagnostics"]
    slurm = summary["slurm"]
    config = summary["config"]
    lines = [
        "# FermiNet PBC-HF Pretrain Benchmark Summary",
        "",
        f"Experiment: `{summary['experiment_name']}`",
        f"Created: `{summary['created_at']}`",
        f"Row source: `{summary['row_source']}`",
        "",
        "## Runtime",
        "",
        f"- Job ID: `{summary.get('job_id') or 'n/a'}`",
        f"- Partition: `{slurm.get('partition', 'n/a')}`",
        f"- Node: `{slurm.get('selected_node', 'n/a')}`",
        f"- GPU model: `{slurm.get('gpu_model', 'n/a')}`",
        f"- GRES: `{slurm.get('gres', 'n/a')}`",
        f"- Elapsed seconds: {_format_optional(runtime.get('elapsed_seconds'))}",
        f"- Seconds per pretrain row: {_format_optional(runtime.get('seconds_per_pretrain_row'))}",
        "",
        "## Config",
        "",
        f"- Pretrain method: `{config.get('pretrain_method')}`",
        f"- Target backend: `{config.get('pretrain_target_backend')}`",
        f"- Basis: `{config.get('pretrain_basis')}`",
        f"- Image cutoff: `{config.get('pretrain_image_cutoff')}`",
        f"- Batch size: `{config.get('pretrain_batch_size')}`",
        f"- Iterations: `{config.get('pretrain_iterations')}`",
        f"- Target chunk size: `{config.get('pretrain_target_chunk_size')}`",
        f"- Precision profile: `{config.get('precision_profile')}`",
        f"- X64 enabled: `{config.get('x64_enabled')}`",
        "",
        "## Pretrain",
        "",
        f"- Rows: {stats['rows']}",
        f"- First/last step: {stats['first_step']} / {stats['last_step']}",
        f"- First loss: {stats['first_loss']:.6g}",
        f"- Last loss: {stats['last_loss']:.6g}",
        f"- Minimum loss: {stats['min_loss']:.6g} at step {stats['min_loss_step']}",
        f"- Loss delta: {stats['loss_delta']:.6g}",
        f"- Loss drop fraction: {_format_optional(stats['loss_drop_fraction'])}",
        f"- First-10 loss mean: {_format_optional(stats['first10_loss_mean'])}",
        f"- Last-10 loss mean: {_format_optional(stats['last10_loss_mean'])}",
        f"- Mean pmove: {stats['pmove_mean']:.6f}",
        f"- Pmove range: [{stats['pmove_min']:.6f}, {stats['pmove_max']:.6f}]",
        f"- Finite checks: `{stats['finite_checks_passed']}`",
        "",
        "## Timing",
        "",
        f"- Warmup rows excluded: {timing['warmup_steps_excluded']}",
        f"- Steady rows: {timing['steady_rows']}",
        f"- Mean steady step seconds: {timing['steady_mean_step_seconds']:.6f}",
        f"- Median steady step seconds: {timing['steady_median_step_seconds']:.6f}",
        f"- P90 steady step seconds: {timing['steady_p90_step_seconds']:.6f}",
        f"- Mean target eval seconds: {timing['steady_mean_target_eval_seconds']:.6f}",
        f"- Median target eval seconds: {timing['steady_median_target_eval_seconds']:.6f}",
        f"- P90 target eval seconds: {timing['steady_p90_target_eval_seconds']:.6f}",
        f"- Mean target transfer seconds: {timing['steady_mean_target_transfer_seconds']:.6f}",
        f"- Mean JAX update seconds: {timing['steady_mean_jax_update_seconds']:.6f}",
        f"- Median JAX update seconds: {timing['steady_median_jax_update_seconds']:.6f}",
        f"- P90 JAX update seconds: {timing['steady_p90_jax_update_seconds']:.6f}",
        f"- Target eval fraction: {_format_optional(timing.get('steady_target_eval_fraction'))}",
        f"- Target transfer fraction: {_format_optional(timing.get('steady_target_transfer_fraction'))}",
        f"- JAX update fraction: {_format_optional(timing.get('steady_jax_update_fraction'))}",
        f"- Step-0 seconds: {timing['compile_row']['step_seconds']:.6f}",
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
    timing = summary["timing"]
    print(f"experiment: {summary['experiment_name']}")
    print(f"rows: {stats['rows']}")
    print(f"loss_first: {stats['first_loss']:.12g}")
    print(f"loss_last: {stats['last_loss']:.12g}")
    print(f"pmove_mean: {stats['pmove_mean']:.12g}")
    print(f"steady_step_seconds: {timing['steady_mean_step_seconds']:.12g}")
    print(f"steady_target_eval_seconds: {timing['steady_mean_target_eval_seconds']:.12g}")
    print(f"steady_jax_update_seconds: {timing['steady_mean_jax_update_seconds']:.12g}")
    print(f"summary_json: {output_json}")
    print(f"summary_md: {output_md}")


if __name__ == "__main__":
    raise SystemExit(main())
