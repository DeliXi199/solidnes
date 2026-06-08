#!/usr/bin/env python3
"""Compare FermiNet-QKV and fused-QKV routes for each 0112 learning rate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROUTE_STYLE = {
    "ferminet": ("#1f77b4", "FermiNet QKV"),
    "fused_qkv": ("#ff7f0e", "Fused QKV"),
}

LR_LABELS = {
    0.02: "lr=2e-2",
    0.01: "lr=1e-2",
    0.005: "lr=5e-3",
}


def _read_timeseries(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | int | str] = {}
            for key, value in row.items():
                if key in {"route", "learning_rate_label"}:
                    parsed[key] = value
                elif key in {"step", "segment"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value) if value else float("nan")
            rows.append(parsed)
    return rows


def _group_rows(rows: list[dict[str, float | int | str]]) -> dict[tuple[float, str], list[dict[str, float | int | str]]]:
    grouped: dict[tuple[float, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (float(row["learning_rate"]), str(row["route"]))
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["step"]))
    return grouped


def _series(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)


def _steps(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    return np.asarray([int(row["step"]) for row in rows], dtype=int)


def _rolling_mean(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    if min_periods is None:
        min_periods = window
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(valid.astype(int))])
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.size):
        start = max(0, i + 1 - window)
        count = ccnt[i + 1] - ccnt[start]
        if count >= min_periods:
            out[i] = (csum[i + 1] - csum[start]) / count
    return out


def _windowed_rolled(
    rows: list[dict[str, float | int | str]],
    key: str,
    start_step: int,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    steps = _steps(rows)
    values = _series(rows, key)
    rolled = _rolling_mean(values, rolling_window)
    mask = (steps >= start_step) & np.isfinite(rolled)
    return steps[mask], rolled[mask]


def _set_rolling_ylim(ax: plt.Axes, series: list[np.ndarray], min_padding: float) -> None:
    finite = [values[np.isfinite(values)] for values in series if np.isfinite(values).any()]
    if not finite:
        return
    values = np.concatenate(finite)
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * 0.08, min_padding)
    ax.set_ylim(low - padding, high + padding)


def _save(fig: plt.Figure, output_base: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".png", ".svg"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_energy_gap_spin(
    output_base: Path,
    grouped: dict[tuple[float, str], list[dict[str, float | int | str]]],
    learning_rate: float,
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    panels = [
        ("e0", "ground energy E0 (Ha)", 0.002),
        ("e1", "excited energy E1 (Ha)", 0.002),
        ("gap_ev", "gap E1 - E0 (eV)", 0.04),
        ("spin", "mean <S^2>", 0.0002),
    ]
    for ax, (key, ylabel, padding) in zip(axes, panels):
        rolled_series: list[np.ndarray] = []
        for route in ("ferminet", "fused_qkv"):
            rows = grouped[(learning_rate, route)]
            steps, rolled = _windowed_rolled(rows, key, start_step, rolling_window)
            rolled_series.append(rolled)
            color, label = ROUTE_STYLE[route]
            ax.plot(steps, rolled, color=color, linewidth=1.9, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _set_rolling_ylim(ax, rolled_series, min_padding=padding)
    axes[-1].set_xlim(start_step, end_step)
    axes[-1].set_xlabel("step")
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    fig.suptitle(
        f"{LR_LABELS[learning_rate]}: QKV route comparison, steps "
        f"{start_step}-{end_step} (trailing {rolling_window}-step mean)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_base)


def _plot_gap(
    output_base: Path,
    grouped: dict[tuple[float, str], list[dict[str, float | int | str]]],
    learning_rate: float,
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    rolled_series: list[np.ndarray] = []
    for route in ("ferminet", "fused_qkv"):
        rows = grouped[(learning_rate, route)]
        steps, rolled = _windowed_rolled(rows, "gap_ev", start_step, rolling_window)
        rolled_series.append(rolled)
        color, label = ROUTE_STYLE[route]
        ax.plot(steps, rolled, color=color, linewidth=2.0, label=label)
    ax.set_title(
        f"{LR_LABELS[learning_rate]}: QKV route gap comparison, "
        f"steps {start_step}-{end_step}"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("gap E1 - E0 (eV)")
    ax.set_xlim(start_step, end_step)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2, fontsize=8, loc="best")
    _set_rolling_ylim(ax, rolled_series, min_padding=0.04)
    fig.tight_layout()
    return _save(fig, output_base)


def _last_window_mean(rows: list[dict[str, float | int | str]], key: str, window: int) -> float:
    steps = _steps(rows)
    values = _series(rows, key)
    mask = steps >= steps.max() - window + 1
    return float(np.nanmean(values[mask]))


def _summarize(
    grouped: dict[tuple[float, str], list[dict[str, float | int | str]]],
    rolling_window: int,
    start_step: int,
    end_step: int,
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for lr in (0.02, 0.01, 0.005):
        row: dict[str, float | int | str] = {
            "learning_rate": lr,
            "learning_rate_label": LR_LABELS[lr],
            "start_step": start_step,
            "end_step": end_step,
        }
        for route in ("ferminet", "fused_qkv"):
            rows = grouped[(lr, route)]
            prefix = route
            for key in ("e0", "e1", "gap_ev", "spin", "pmove"):
                row[f"{prefix}_last{rolling_window}_{key}_mean"] = _last_window_mean(rows, key, rolling_window)
        for key in ("e0", "e1", "gap_ev", "spin", "pmove"):
            row[f"fused_minus_ferminet_last{rolling_window}_{key}_mean"] = (
                float(row[f"fused_qkv_last{rolling_window}_{key}_mean"])
                - float(row[f"ferminet_last{rolling_window}_{key}_mean"])
            )
        summaries.append(row)
    return summaries


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summaries_by_window: dict[str, list[dict[str, float | int | str]]],
    plot_paths: list[Path],
    rolling_window: int,
) -> None:
    def fmt(value: object, digits: int = 6) -> str:
        return f"{float(value):.{digits}f}"

    lines = [
        "# 0112 Learning-Rate QKV Route Comparison",
        "",
        "Compared `ferminet` QKV and `fused_qkv` at each learning rate. "
        "Plots use trailing 1000-step means and keep each learning rate in a separate figure.",
        "",
    ]
    for window_name, summaries in summaries_by_window.items():
        lines.extend(
            [
                f"## {window_name}",
                "",
                "| Learning rate | ΔE0 fused-ferminet Ha | ΔE1 fused-ferminet Ha | Δgap fused-ferminet eV | Δspin fused-ferminet |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{float(row['learning_rate']):g}",
                        fmt(row[f"fused_minus_ferminet_last{rolling_window}_e0_mean"]),
                        fmt(row[f"fused_minus_ferminet_last{rolling_window}_e1_mean"]),
                        fmt(row[f"fused_minus_ferminet_last{rolling_window}_gap_ev_mean"], 4),
                        fmt(row[f"fused_minus_ferminet_last{rolling_window}_spin_mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Generated Plots", ""])
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/psiformer/0112_attention_qkv_spin_beta10_damp1e3_lr_sweep_continue_10000/analysis"),
    )
    parser.add_argument("--rolling-window", type=int, default=1000)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        ("after30000", output_dir / "0112_lr_sweep_combined_timeseries.csv", 30000, 39999),
        ("after20000", output_dir / "0112_lr_sweep_after20000_combined_timeseries.csv", 20000, 39999),
    ]
    all_plot_paths: list[Path] = []
    summaries_by_window: dict[str, list[dict[str, float | int | str]]] = {}
    for window_name, input_path, start_step, end_step in inputs:
        grouped = _group_rows(_read_timeseries(input_path))
        summaries = _summarize(grouped, args.rolling_window, start_step, end_step)
        summaries_by_window[window_name] = summaries
        summary_csv = output_dir / f"0112_lr_qkv_route_comparison_{window_name}_summary.csv"
        _write_summary_csv(summary_csv, summaries)
        summary_json = output_dir / f"0112_lr_qkv_route_comparison_{window_name}_summary.json"
        summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True))
        print(f"wrote {summary_csv}")
        print(f"wrote {summary_json}")
        for lr in (0.02, 0.01, 0.005):
            lr_tag = {0.02: "lr2e2", 0.01: "lr1e2", 0.005: "lr5e3"}[lr]
            base = output_dir / (
                f"0112_lr_qkv_route_comparison_{window_name}_{lr_tag}_"
                f"energy_gap_spin_rolling_after{start_step}_window{args.rolling_window}"
            )
            all_plot_paths.extend(
                _plot_energy_gap_spin(base, grouped, lr, start_step, end_step, args.rolling_window)
            )
            gap_base = output_dir / (
                f"0112_lr_qkv_route_comparison_{window_name}_{lr_tag}_"
                f"gap_rolling_after{start_step}_window{args.rolling_window}"
            )
            all_plot_paths.extend(
                _plot_gap(gap_base, grouped, lr, start_step, end_step, args.rolling_window)
            )

    md_path = output_dir / "0112_lr_qkv_route_comparison.md"
    _write_markdown(md_path, summaries_by_window, all_plot_paths, args.rolling_window)
    print(f"wrote {md_path}")
    for path in all_plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
