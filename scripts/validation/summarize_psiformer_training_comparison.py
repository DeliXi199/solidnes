#!/usr/bin/env python
"""Summarize task 0095 PsiFormer native training variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "tasks/psiformer/0095_psiformer_native_training_smoke"
DEFAULT_VARIANTS = ("auto_smoke", "ferminet_b512", "fused_qkv_b512", "fused_qkv_b1024")


def main() -> int:
    args = _parse_args()
    task_root = _resolve_path(args.task_root)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    summary = {
        "task_root": _display_path(task_root),
        "variants": [_variant_summary(task_root, variant) for variant in variants],
    }
    summary["comparisons"] = _comparisons(summary["variants"])
    output_dir = task_root / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "psiformer_training_comparison.json"
    md_path = output_dir / "psiformer_training_comparison.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(summary), encoding="utf-8")
    print("psiformer_training_comparison: ok")
    print(f"summary_json: {_display_path(json_path)}")
    print(f"summary_markdown: {_display_path(md_path)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    return parser.parse_args()


def _variant_summary(task_root: Path, variant: str) -> dict[str, Any]:
    run_root = task_root / "runs" / variant
    validation_dir = run_root / "results/validation"
    checkpoint_dir = run_root / "results/checkpoints"
    runtime = _read_json(validation_dir / "ferminet_train_runtime.json")
    native_summary = _read_json(validation_dir / "native_ferminet_excited_summary.json")
    rows = _read_train_stats(checkpoint_dir / "train_stats.csv")
    result: dict[str, Any] = {
        "variant": variant,
        "run_root": _display_path(run_root),
        "status": "completed" if rows else "missing_train_stats",
        "rows": len(rows),
        "final_energy": _float_or_none(rows[-1].get("energy")) if rows else None,
        "mean_pmove": _mean([_float_or_none(row.get("pmove")) for row in rows]),
        "runtime": runtime,
        "native_summary": native_summary,
    }
    if runtime:
        result.update(
            {
                "attention_configured": runtime.get("psiformer_attention_configured"),
                "attention_resolved": runtime.get("psiformer_attention_resolved"),
                "batch_size": runtime.get("batch_size"),
                "iterations": runtime.get("iterations"),
                "elapsed_seconds": runtime.get("elapsed_seconds"),
                "seconds_per_iteration": runtime.get("seconds_per_iteration"),
            }
        )
    if native_summary:
        result.update(
            {
                "summary_status": native_summary.get("status"),
                "final_state_energy": native_summary.get("final_state_energy"),
                "final_training_gap_ev": native_summary.get("final_training_gap_ev"),
                "final_symmetric_overlap_matrix": native_summary.get(
                    "final_symmetric_overlap_matrix"
                ),
            }
        )
    return result


def _comparisons(variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {variant["variant"]: variant for variant in variants}
    upstream = by_name.get("ferminet_b512", {})
    fused = by_name.get("fused_qkv_b512", {})
    result: dict[str, Any] = {}
    upstream_time = _float_or_none(upstream.get("seconds_per_iteration"))
    fused_time = _float_or_none(fused.get("seconds_per_iteration"))
    if upstream_time and fused_time:
        result["b512_seconds_per_iteration_speedup"] = upstream_time / fused_time
        result["b512_elapsed_speedup"] = (
            _float_or_none(upstream.get("elapsed_seconds"))
            / _float_or_none(fused.get("elapsed_seconds"))
        )
    larger = by_name.get("fused_qkv_b1024", {})
    larger_time = _float_or_none(larger.get("seconds_per_iteration"))
    if fused_time and larger_time:
        result["fused_b512_to_b1024_seconds_per_iteration_ratio"] = (
            larger_time / fused_time
        )
    return result


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PsiFormer Native Training Comparison",
        "",
        f"Task root: `{summary['task_root']}`",
        "",
        "| Variant | Status | Attention | Batch | Iter | Rows | Elapsed s | s/iter | Final E | Mean pmove |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in summary["variants"]:
        lines.append(
            "| {variant} | {status} | {attention} | {batch} | {iterations} | "
            "{rows} | {elapsed} | {seconds} | {energy} | {pmove} |".format(
                variant=variant["variant"],
                status=variant.get("status"),
                attention=variant.get("attention_resolved")
                or variant.get("attention_configured")
                or "",
                batch=_fmt(variant.get("batch_size"), integer=True),
                iterations=_fmt(variant.get("iterations"), integer=True),
                rows=_fmt(variant.get("rows"), integer=True),
                elapsed=_fmt(variant.get("elapsed_seconds")),
                seconds=_fmt(variant.get("seconds_per_iteration")),
                energy=_fmt(variant.get("final_energy")),
                pmove=_fmt(variant.get("mean_pmove")),
            )
        )
    lines.extend(["", "## Comparisons", ""])
    comparisons = summary.get("comparisons", {})
    if not comparisons:
        lines.append("No matched timing comparison is available yet.")
    else:
        for key, value in comparisons.items():
            lines.append(f"- `{key}`: `{_fmt(value)}`")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_train_stats(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _fmt(value: Any, *, integer: bool = False) -> str:
    if value is None:
        return ""
    if integer:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
