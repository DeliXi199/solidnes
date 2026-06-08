#!/usr/bin/env python3
"""Plot the 0113 PsiFormer eta/tau fresh-start sweep.

The six 0113 runs share the FermiNet-QKV route, beta=10 spin penalty, and
KFAC damping=1e-3. They differ only in eta0 and tau for the learning-rate
schedule. This script compares the post-burn-in iteration traces.
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
class RunSpec:
    run: str
    eta0: float
    tau: int
    label: str
    checkpoint_dir: Path


def _default_run_specs(repo_root: Path) -> list[RunSpec]:
    task = repo_root / "tasks/psiformer/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000"

    def spec(run: str, eta0: float, tau: int) -> RunSpec:
        return RunSpec(
            run=run,
            eta0=eta0,
            tau=tau,
            label=f"eta0={eta0:g}, tau={tau}",
            checkpoint_dir=task / "runs" / run / "ferminet_merge_none/results/checkpoints",
        )

    return [
        spec("eta1e2_tau10000", 0.01, 10000),
        spec("eta1e2_tau15000", 0.01, 15000),
        spec("eta1e2_tau20000", 0.01, 20000),
        spec("eta2e2_tau10000", 0.02, 10000),
        spec("eta2e2_tau15000", 0.02, 15000),
        spec("eta2e2_tau20000", 0.02, 20000),
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
    stats_path = spec.checkpoint_dir / "train_stats.csv"
    energy_path = spec.checkpoint_dir / "energy_matrix.npy"
    stats = _read_train_stats(stats_path)
    energies = _load_npy_stream(energy_path)
    warnings: list[str] = []
    if len(energies) != len(stats):
        warnings.append(
            f"{spec.run}: energy_matrix records={len(energies)}, "
            f"train_stats rows={len(stats)}; using aligned prefix"
        )

    rows: list[dict[str, float | int | str]] = []
    usable_energies = energies[: len(stats)]
    for i, row in enumerate(stats):
        out: dict[str, float | int | str] = {
            "run": spec.run,
            "eta0": spec.eta0,
            "tau": spec.tau,
            "parameter_label": spec.label,
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


def _style_for_specs(specs: list[RunSpec]) -> dict[str, tuple[str, str, str]]:
    colors = {
        (0.01, 10000): "#1f77b4",
        (0.01, 15000): "#2ca02c",
        (0.01, 20000): "#17becf",
        (0.02, 10000): "#d62728",
        (0.02, 15000): "#ff7f0e",
        (0.02, 20000): "#9467bd",
    }
    linestyles = {10000: "-", 15000: "--", 20000: "-."}
    return {
        spec.run: (
            colors[(spec.eta0, spec.tau)],
            linestyles[spec.tau],
            spec.label,
        )
        for spec in specs
    }


def _set_rolling_ylim(ax: plt.Axes, series: list[np.ndarray], min_padding: float) -> None:
    finite_parts = [values[np.isfinite(values)] for values in series if np.isfinite(values).any()]
    if not finite_parts:
        return
    values = np.concatenate(finite_parts)
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * 0.08, min_padding)
    ax.set_ylim(low - padding, high + padding)


def _save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".png", ".svg"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_metric(
    output_base: Path,
    grouped: dict[str, list[dict[str, float | int | str]]],
    specs: list[RunSpec],
    metric_key: str,
    ylabel: str,
    title_metric: str,
    start_step: int,
    end_step: int,
    rolling_window: int,
    min_padding: float,
) -> list[Path]:
    style = _style_for_specs(specs)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.3))
    rolled_series: list[np.ndarray] = []
    for spec in specs:
        rows = grouped[spec.run]
        steps, rolled = _windowed_rolled(rows, metric_key, start_step, rolling_window)
        rolled_series.append(rolled)
        color, linestyle, label = style[spec.run]
        ax.plot(steps, rolled, color=color, linestyle=linestyle, linewidth=1.8, label=label)

    ax.set_title(
        f"0113 eta/tau sweep: {title_metric}, steps {start_step}-{end_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(start_step, end_step)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2, fontsize=8, loc="best")
    _set_rolling_ylim(ax, rolled_series, min_padding=min_padding)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def _plot_overview(
    output_base: Path,
    grouped: dict[str, list[dict[str, float | int | str]]],
    specs: list[RunSpec],
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    style = _style_for_specs(specs)
    panels = [
        ("e0", "ground energy E0 (Ha)", "ground-state energy E0", 0.002),
        ("e1", "excited energy E1 (Ha)", "excited-state energy E1", 0.002),
        ("spin", "mean <S^2>", "spin", 0.0002),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (metric_key, ylabel, _title, min_padding) in zip(axes, panels):
        rolled_series: list[np.ndarray] = []
        for spec in specs:
            rows = grouped[spec.run]
            steps, rolled = _windowed_rolled(rows, metric_key, start_step, rolling_window)
            rolled_series.append(rolled)
            color, linestyle, label = style[spec.run]
            ax.plot(steps, rolled, color=color, linestyle=linestyle, linewidth=1.6, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _set_rolling_ylim(ax, rolled_series, min_padding=min_padding)
    axes[-1].set_xlabel("step")
    axes[-1].set_xlim(start_step, end_step)
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    fig.suptitle(
        f"0113 eta/tau sweep, steps {start_step}-{end_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, output_base)


def _write_timeseries(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    keys = [
        "run",
        "eta0",
        "tau",
        "parameter_label",
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
        "post_step_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_for_rows(
    rows: list[dict[str, float | int | str]],
    rolling_window: int,
    start_step: int,
) -> dict[str, float | int | str]:
    steps = _steps(rows)
    after_mask = steps >= start_step
    last_mask = steps >= steps.max() - rolling_window + 1
    first_after_mask = (steps >= start_step) & (steps < start_step + rolling_window)
    out: dict[str, float | int | str] = {
        "run": rows[0]["run"],
        "eta0": rows[0]["eta0"],
        "tau": rows[0]["tau"],
        "parameter_label": rows[0]["parameter_label"],
        "start_step": int(steps.min()),
        "end_step": int(steps.max()),
        "num_rows": int(len(rows)),
        "analysis_start_step": start_step,
    }
    for key in ("energy", "e0", "e1", "gap_hartree", "gap_ev", "spin", "spin_state_0", "spin_state_1", "pmove"):
        values = _series(rows, key)
        out[f"after{start_step}_{key}_mean"] = float(np.nanmean(values[after_mask]))
        out[f"first_after{start_step}_{rolling_window}_{key}_mean"] = float(np.nanmean(values[first_after_mask]))
        out[f"last{rolling_window}_{key}_mean"] = float(np.nanmean(values[last_mask]))
        out[f"last{rolling_window}_{key}_std"] = float(np.nanstd(values[last_mask], ddof=1))
    out["final_energy"] = float(rows[-1].get("energy", float("nan")))
    out["final_e0"] = float(rows[-1].get("e0", float("nan")))
    out["final_e1"] = float(rows[-1].get("e1", float("nan")))
    out["final_gap_ev"] = float(rows[-1].get("gap_ev", float("nan")))
    out["final_spin"] = float(rows[-1].get("spin", float("nan")))
    return out


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summaries: list[dict[str, float | int | str]],
    warnings: list[str],
    plot_paths: list[Path],
    rolling_window: int,
    start_step: int,
) -> None:
    def fmt(value: object, digits: int = 6) -> str:
        return f"{float(value):.{digits}f}"

    lines = [
        "# 0113 Eta/Tau Sweep Comparison",
        "",
        f"Six eta0/tau settings are compared after step {start_step}. "
        f"Plots use trailing {rolling_window}-step means.",
        "",
        f"Final-window statistics use steps {30000 - rolling_window}-{29999}.",
        "",
        "| Setting | Rows | E0 mean Ha | E1 mean Ha | Gap eV | Spin mean | pmove |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda r: (float(r["eta0"]), int(r["tau"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["parameter_label"]),
                    str(row["num_rows"]),
                    fmt(row[f"last{rolling_window}_e0_mean"]),
                    fmt(row[f"last{rolling_window}_e1_mean"]),
                    fmt(row[f"last{rolling_window}_gap_ev_mean"], 4),
                    fmt(row[f"last{rolling_window}_spin_mean"]),
                    fmt(row[f"last{rolling_window}_pmove_mean"], 4),
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
        default=Path("tasks/psiformer/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000/analysis"),
    )
    parser.add_argument("--rolling-window", type=int, default=1000)
    parser.add_argument("--start-step", type=int, default=5000)
    parser.add_argument("--end-step", type=int, default=29999)
    parser.add_argument("--prefix", default="0113_eta_tau_sweep")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _default_run_specs(repo_root)
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    all_rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    for spec in specs:
        rows, spec_warnings = _rows_for_spec(spec)
        grouped[spec.run] = rows
        all_rows.extend(rows)
        warnings.extend(spec_warnings)

    summaries = [
        _summary_for_rows(rows, args.rolling_window, args.start_step)
        for rows in grouped.values()
        if rows
    ]

    prefix = output_dir / args.prefix
    timeseries_path = prefix.with_name(prefix.name + "_combined_timeseries.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")
    summary_json_path = prefix.with_name(prefix.name + "_summary.json")
    markdown_path = prefix.with_name(prefix.name + "_comparison.md")

    _write_timeseries(timeseries_path, all_rows)
    _write_summary_csv(summary_csv_path, summaries)
    summary_json_path.write_text(json.dumps({"summary": summaries, "warnings": warnings}, indent=2, sort_keys=True))

    metric_specs = [
        ("e0", "ground energy E0 (Ha)", "ground-state energy E0", "ground_state", 0.002),
        ("e1", "excited energy E1 (Ha)", "excited-state energy E1", "excited_state", 0.002),
        ("gap_ev", "gap E1 - E0 (eV)", "energy gap E1 - E0", "gap", 0.04),
        ("spin", "mean <S^2>", "spin", "spin", 0.0002),
    ]
    plot_paths: list[Path] = []
    for metric_key, ylabel, title_metric, filename_metric, min_padding in metric_specs:
        output_base = prefix.with_name(
            f"{prefix.name}_{filename_metric}_rolling_after{args.start_step}_window{args.rolling_window}"
        )
        plot_paths.extend(
            _plot_metric(
                output_base,
                grouped,
                specs,
                metric_key,
                ylabel,
                title_metric,
                args.start_step,
                args.end_step,
                args.rolling_window,
                min_padding,
            )
        )
    overview_base = prefix.with_name(
        f"{prefix.name}_ground_excited_spin_rolling_after{args.start_step}_window{args.rolling_window}"
    )
    plot_paths.extend(
        _plot_overview(
            overview_base,
            grouped,
            specs,
            args.start_step,
            args.end_step,
            args.rolling_window,
        )
    )

    _write_markdown(markdown_path, summaries, warnings, plot_paths, args.rolling_window, args.start_step)

    print(f"Wrote {timeseries_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {markdown_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
