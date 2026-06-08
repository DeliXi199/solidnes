#!/usr/bin/env python3
"""Plot the PsiFormer damping continuation sweep.

The 0110/0111 damping runs all branch from the completed 0108 step-29999
checkpoints.  This script compares only the continuation interval, steps
30000 through 39999.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class Segment:
    checkpoint_dir: Path


@dataclass(frozen=True)
class RunSpec:
    route: str
    damping: float
    label: str
    segments: tuple[Segment, ...]


def _default_run_specs(repo_root: Path) -> list[RunSpec]:
    task_0110 = repo_root / "tasks/psiformer/0110_attention_qkv_spin_beta10_damp2e3_continue_10000"
    task_0111 = repo_root / "tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000"
    return [
        RunSpec(
            "ferminet",
            0.002,
            "d=0.002",
            (Segment(task_0110 / "runs/ferminet_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "fused_qkv",
            0.002,
            "d=0.002",
            (Segment(task_0110 / "runs/fused_qkv_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "ferminet",
            0.003,
            "d=0.003",
            (Segment(task_0111 / "runs/damp3e3/ferminet_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "fused_qkv",
            0.003,
            "d=0.003",
            (Segment(task_0111 / "runs/damp3e3/fused_qkv_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "ferminet",
            0.005,
            "d=0.005",
            (Segment(task_0111 / "runs/damp5e3/ferminet_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "fused_qkv",
            0.005,
            "d=0.005",
            (
                Segment(task_0111 / "runs/damp5e3/fused_qkv_merge_none/results/checkpoints"),
                Segment(task_0111 / "runs/damp5e3/fused_qkv_merge_none_resume36433/results/checkpoints"),
            ),
        ),
        RunSpec(
            "ferminet",
            0.01,
            "d=0.01",
            (Segment(task_0111 / "runs/damp1e2/ferminet_merge_none/results/checkpoints"),),
        ),
        RunSpec(
            "fused_qkv",
            0.01,
            "d=0.01",
            (Segment(task_0111 / "runs/damp1e2/fused_qkv_merge_none/results/checkpoints"),),
        ),
    ]


def _read_train_stats(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = float("nan")
                elif key == "step":
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _load_npy_stream(path: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if not path.exists() or path.stat().st_size == 0:
        return arrays
    with path.open("rb") as f:
        while True:
            try:
                arrays.append(np.load(f, allow_pickle=False))
            except (EOFError, ValueError):
                break
    return arrays


def _rows_for_spec(spec: RunSpec) -> tuple[list[dict[str, float | int | str]], list[str]]:
    rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    for segment_index, segment in enumerate(spec.segments):
        stats_path = segment.checkpoint_dir / "train_stats.csv"
        energy_path = segment.checkpoint_dir / "energy_matrix.npy"
        stats = _read_train_stats(stats_path)
        energies = _load_npy_stream(energy_path)
        if len(energies) != len(stats):
            warnings.append(
                f"{spec.route} {spec.label} segment {segment_index}: "
                f"energy_matrix records={len(energies)}, train_stats rows={len(stats)}; "
                "using train_stats row count for alignment"
            )
        usable_energies = energies[: len(stats)]
        for i, row in enumerate(stats):
            out: dict[str, float | int | str] = {
                "route": spec.route,
                "damping": spec.damping,
                "damping_label": spec.label,
                "segment": segment_index,
                **row,
            }
            if i < len(usable_energies):
                energy = np.asarray(usable_energies[i], dtype=float).reshape(-1)
                if energy.size >= 2:
                    sorted_energy = np.sort(energy[:2])
                    out["e0"] = float(sorted_energy[0])
                    out["e1"] = float(sorted_energy[1])
                    out["gap_hartree"] = float(sorted_energy[1] - sorted_energy[0])
                    out["gap_ev"] = float((sorted_energy[1] - sorted_energy[0]) * HARTREE_TO_EV)
                else:
                    out["e0"] = float("nan")
                    out["e1"] = float("nan")
                    out["gap_hartree"] = float("nan")
                    out["gap_ev"] = float("nan")
            else:
                out["e0"] = float("nan")
                out["e1"] = float("nan")
                out["gap_hartree"] = float("nan")
                out["gap_ev"] = float("nan")
            rows.append(out)
    by_step: dict[int, dict[str, float | int | str]] = {}
    for row in rows:
        by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)], warnings


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


def _series(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)


def _steps(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    return np.asarray([int(row["step"]) for row in rows], dtype=int)


def _write_timeseries(path: Path, all_rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(all_rows)
    keys = [
        "route",
        "damping",
        "damping_label",
        "segment",
        "step",
        "energy",
        "ewmean",
        "ewvar",
        "e0",
        "e1",
        "gap_hartree",
        "gap_ev",
        "spin",
        "spin_penalty",
        "spin_state_0",
        "spin_state_1",
        "pmove",
        "step_seconds",
        "mcmc_seconds",
        "optimizer_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_for_rows(rows: list[dict[str, float | int | str]], rolling_window: int) -> dict[str, float | int | str]:
    steps = _steps(rows)
    tail_mask = steps >= (steps.max() - rolling_window + 1)
    head_mask = steps <= (steps.min() + rolling_window - 1)
    out: dict[str, float | int | str] = {
        "route": rows[0]["route"],
        "damping": rows[0]["damping"],
        "damping_label": rows[0]["damping_label"],
        "start_step": int(steps.min()),
        "end_step": int(steps.max()),
        "num_rows": int(len(rows)),
    }
    for key in ("energy", "e0", "e1", "gap_hartree", "gap_ev", "spin", "spin_state_0", "spin_state_1", "pmove", "step_seconds"):
        values = _series(rows, key)
        tail = values[tail_mask]
        head = values[head_mask]
        out[f"first{rolling_window}_{key}_mean"] = float(np.nanmean(head))
        out[f"last{rolling_window}_{key}_mean"] = float(np.nanmean(tail))
        out[f"last{rolling_window}_{key}_std"] = float(np.nanstd(tail, ddof=1))
        out[f"delta_last_minus_first_{key}"] = float(np.nanmean(tail) - np.nanmean(head))
    out["final_energy"] = float(rows[-1].get("energy", float("nan")))
    out["final_e0"] = float(rows[-1].get("e0", float("nan")))
    out["final_e1"] = float(rows[-1].get("e1", float("nan")))
    out["final_gap_ev"] = float(rows[-1].get("gap_ev", float("nan")))
    out["final_spin"] = float(rows[-1].get("spin", float("nan")))
    return out


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _damping_style() -> dict[float, tuple[str, str]]:
    return {
        0.002: ("#1f77b4", "d=2e-3"),
        0.003: ("#2ca02c", "d=3e-3"),
        0.005: ("#ff7f0e", "d=5e-3"),
        0.01: ("#d62728", "d=1e-2"),
    }


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


def _set_rolling_ylim(ax: plt.Axes, series: list[np.ndarray], min_padding: float = 0.002) -> None:
    finite_parts = [values[np.isfinite(values)] for values in series if np.isfinite(values).any()]
    if not finite_parts:
        return
    values = np.concatenate(finite_parts)
    low = float(values.min())
    high = float(values.max())
    span = high - low
    padding = max(span * 0.08, min_padding)
    ax.set_ylim(low - padding, high + padding)


def _save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    paths = []
    for suffix in (".png", ".svg"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_route_energy_gap_spin(
    output_base: Path,
    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]],
    route: str,
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    style = _damping_style()
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    panels = [
        ("e0", "ground energy E0 (Ha)", 0.002),
        ("e1", "excited energy E1 (Ha)", 0.002),
        ("gap_ev", "gap E1 - E0 (eV)", 0.04),
        ("spin", "mean <S^2>", 0.0002),
    ]
    for ax, (key, ylabel, padding) in zip(axes, panels):
        rolled_series: list[np.ndarray] = []
        for damping, (color, label) in style.items():
            rows = grouped.get((route, damping), [])
            if not rows:
                continue
            steps, rolled = _windowed_rolled(rows, key, start_step, rolling_window)
            rolled_series.append(rolled)
            ax.plot(steps, rolled, color=color, linewidth=1.8, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _set_rolling_ylim(ax, rolled_series, min_padding=padding)
    axes[-1].set_xlabel("step")
    axes[-1].set_xlim(start_step, end_step)
    axes[0].legend(ncol=4, fontsize=8, loc="best")
    fig.suptitle(
        f"{route} damping sweep, steps {start_step}-{end_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, output_base)


def _plot_route_gap(
    output_base: Path,
    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]],
    route: str,
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    style = _damping_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    rolled_series: list[np.ndarray] = []
    for damping, (color, label) in style.items():
        rows = grouped.get((route, damping), [])
        if not rows:
            continue
        steps, rolled = _windowed_rolled(rows, "gap_ev", start_step, rolling_window)
        rolled_series.append(rolled)
        ax.plot(steps, rolled, color=color, linewidth=1.9, label=label)
    ax.set_title(
        f"{route} gap, steps {start_step}-{end_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("gap E1 - E0 (eV)")
    ax.set_xlim(start_step, end_step)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=4, fontsize=8, loc="best")
    _set_rolling_ylim(ax, rolled_series, min_padding=0.04)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def _write_markdown(
    path: Path,
    summaries: list[dict[str, float | int | str]],
    warnings: list[str],
    plot_paths: list[Path],
    rolling_window: int,
) -> None:
    def fmt(value: object, digits: int = 6) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    rows = sorted(summaries, key=lambda r: (str(r["route"]), float(r["damping"])))
    lines = [
        "# 0111 Damping Sweep Comparison",
        "",
        "Compared continuation steps 30000 through 39999.  All runs restore from the completed 0108 step-29999 checkpoints and keep `learning_rate=0.05`, `spin_penalty=10.0`, and `norm_constraint=0.001`; only KFAC damping changes.",
        "",
        f"Statistics below use the final trailing {rolling_window}-step window.",
        "",
        "| Route | Damping | Rows | E0 mean Ha | E1 mean Ha | Gap eV | Spin mean | pmove | step s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["route"]),
                    f"{float(row['damping']):g}",
                    str(row["num_rows"]),
                    fmt(row["last1000_e0_mean"]),
                    fmt(row["last1000_e1_mean"]),
                    fmt(row["last1000_gap_ev_mean"], 4),
                    fmt(row["last1000_spin_mean"], 6),
                    fmt(row["last1000_pmove_mean"], 4),
                    fmt(row["last1000_step_seconds_mean"], 3),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Generated Plots", ""])
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    if warnings:
        lines.extend(["", "## Data Notes", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "tasks/psiformer/0111_attention_qkv_spin_beta10_damping_sweep_continue_10000/analysis"
        ),
    )
    parser.add_argument("--rolling-window", type=int, default=1000)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    all_rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    for spec in _default_run_specs(repo_root):
        rows, spec_warnings = _rows_for_spec(spec)
        grouped[(spec.route, spec.damping)] = rows
        all_rows.extend(rows)
        warnings.extend(spec_warnings)

    summaries = [
        _summary_for_rows(rows, rolling_window=args.rolling_window)
        for rows in grouped.values()
        if rows
    ]

    prefix = output_dir / "0111_damping_sweep"
    timeseries_path = prefix.with_name(prefix.name + "_combined_timeseries.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")
    summary_json_path = prefix.with_name(prefix.name + "_summary.json")
    _write_timeseries(timeseries_path, all_rows)
    _write_summary_csv(summary_csv_path, summaries)
    summary_json_path.write_text(json.dumps({"summary": summaries, "warnings": warnings}, indent=2, sort_keys=True))

    routes = ["ferminet", "fused_qkv"]
    plot_paths: list[Path] = []
    for route in routes:
        for start_step in (30000, 35000):
            suffix = f"{route}_energy_gap_spin_rolling_after{start_step}_window{args.rolling_window}"
            energy_base = prefix.with_name(prefix.name + f"_{suffix}")
            plot_paths.extend(
                _plot_route_energy_gap_spin(
                    energy_base,
                    grouped,
                    route,
                    start_step=start_step,
                    end_step=39999,
                    rolling_window=args.rolling_window,
                )
            )
            gap_suffix = f"{route}_gap_rolling_after{start_step}_window{args.rolling_window}"
            gap_base = prefix.with_name(prefix.name + f"_{gap_suffix}")
            plot_paths.extend(
                _plot_route_gap(
                    gap_base,
                    grouped,
                    route,
                    start_step=start_step,
                    end_step=39999,
                    rolling_window=args.rolling_window,
                )
            )

    md_path = prefix.with_name(prefix.name + "_comparison.md")
    _write_markdown(md_path, summaries, warnings, plot_paths, args.rolling_window)

    print(f"wrote {timeseries_path}")
    print(f"wrote {summary_csv_path}")
    print(f"wrote {summary_json_path}")
    print(f"wrote {md_path}")
    for plot_path in plot_paths:
        print(f"wrote {plot_path}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
