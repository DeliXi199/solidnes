#!/usr/bin/env python3
"""Compare eta0 values at fixed tau across the 0113 and 0114 fresh sweeps."""

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
    task_0113 = repo_root / "tasks/psiformer/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000"
    task_0114 = repo_root / "tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000"

    def spec(task: Path, run: str, eta0: float, tau: int) -> RunSpec:
        return RunSpec(
            run=run,
            eta0=eta0,
            tau=tau,
            label=f"eta0={eta0:g}",
            checkpoint_dir=task / "runs" / run / "ferminet_merge_none/results/checkpoints",
        )

    return [
        spec(task_0114, "eta1e3_tau10000", 0.001, 10000),
        spec(task_0114, "eta5e3_tau10000", 0.005, 10000),
        spec(task_0113, "eta1e2_tau10000", 0.01, 10000),
        spec(task_0113, "eta2e2_tau10000", 0.02, 10000),
        spec(task_0114, "eta1e3_tau15000", 0.001, 15000),
        spec(task_0114, "eta5e3_tau15000", 0.005, 15000),
        spec(task_0113, "eta1e2_tau15000", 0.01, 15000),
        spec(task_0113, "eta2e2_tau15000", 0.02, 15000),
        spec(task_0114, "eta1e3_tau20000", 0.001, 20000),
        spec(task_0114, "eta5e3_tau20000", 0.005, 20000),
        spec(task_0113, "eta1e2_tau20000", 0.01, 20000),
        spec(task_0113, "eta2e2_tau20000", 0.02, 20000),
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
    if len(stats) != len(energies):
        warnings.append(
            f"{spec.run}: train_stats rows={len(stats)}, "
            f"energy_matrix records={len(energies)}; aligned prefix used"
        )

    rows: list[dict[str, float | int | str]] = []
    for i, row in enumerate(stats):
        out: dict[str, float | int | str] = {
            "run": spec.run,
            "eta0": spec.eta0,
            "tau": spec.tau,
            "parameter_label": spec.label,
            **row,
        }
        if i < len(energies):
            energy = np.asarray(energies[i], dtype=float).reshape(-1)
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


def _steps(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    return np.asarray([int(row["step"]) for row in rows], dtype=int)


def _series(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)


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


def _rolling_var(
    values: np.ndarray,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> np.ndarray:
    if min_periods is None:
        min_periods = window
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    safe_values = np.where(valid, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(safe_values)])
    csum2 = np.concatenate([[0.0], np.cumsum(safe_values * safe_values)])
    ccnt = np.concatenate([[0], np.cumsum(valid.astype(int))])
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.size):
        start = max(0, i + 1 - window)
        count = ccnt[i + 1] - ccnt[start]
        if count >= min_periods and count > ddof:
            total = csum[i + 1] - csum[start]
            total2 = csum2[i + 1] - csum2[start]
            variance = (total2 - total * total / count) / (count - ddof)
            out[i] = max(float(variance), 0.0)
    return out


def _rolling_stat(values: np.ndarray, window: int, statistic: str) -> np.ndarray:
    if statistic == "mean":
        return _rolling_mean(values, window)
    if statistic == "var":
        return _rolling_var(values, window)
    if statistic == "std":
        return np.sqrt(_rolling_var(values, window))
    if statistic == "abs_delta_mean":
        values = np.asarray(values, dtype=float)
        delta = np.full(values.shape, np.nan, dtype=float)
        finite_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        delta[1:][finite_pairs] = np.abs(values[1:][finite_pairs] - values[:-1][finite_pairs])
        return _rolling_mean(delta, window)
    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def _windowed_rolled(
    rows: list[dict[str, float | int | str]],
    key: str,
    start_step: int,
    rolling_window: int,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    steps = _steps(rows)
    rolled = _rolling_stat(_series(rows, key), rolling_window, statistic)
    mask = (steps >= start_step) & np.isfinite(rolled)
    return steps[mask], rolled[mask]


def _eta_style() -> dict[float, tuple[str, str, str]]:
    return {
        0.001: ("#9467bd", "-.", "eta0=0.001"),
        0.005: ("#1f77b4", "-", "eta0=0.005"),
        0.01: ("#2ca02c", "--", "eta0=0.01"),
        0.02: ("#d62728", ":", "eta0=0.02"),
    }


def _set_rolling_ylim(ax: plt.Axes, series: list[np.ndarray], min_padding: float) -> None:
    finite = [values[np.isfinite(values)] for values in series if np.isfinite(values).any()]
    if not finite:
        return
    values = np.concatenate(finite)
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


def _plot_fixed_tau_panels(
    output_base: Path,
    grouped: dict[tuple[int, float], list[dict[str, float | int | str]]],
    tau: int,
    start_step: int,
    end_step: int,
    rolling_window: int,
    panels: list[tuple[str, str, float]],
    statistic: str,
    title_suffix: str,
) -> list[Path]:
    eta_style = _eta_style()
    fig_height = 2.35 * len(panels) + 1.0
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, fig_height), sharex=True)
    axes_list = np.atleast_1d(axes).tolist()
    for ax, (key, ylabel, padding) in zip(axes_list, panels):
        rolled_series: list[np.ndarray] = []
        for eta0 in (0.001, 0.005, 0.01, 0.02):
            rows = grouped[(tau, eta0)]
            steps, rolled = _windowed_rolled(rows, key, start_step, rolling_window, statistic)
            rolled_series.append(rolled)
            color, linestyle, label = eta_style[eta0]
            ax.plot(steps, rolled, color=color, linestyle=linestyle, linewidth=1.8, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if statistic in {"var", "std", "abs_delta_mean"} or key == "ewvar":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        _set_rolling_ylim(ax, rolled_series, padding)
    axes_list[-1].set_xlim(start_step, end_step)
    axes_list[-1].set_xlabel("step")
    axes_list[0].legend(ncol=4, fontsize=8, loc="best")
    fig.suptitle(
        f"Fixed tau={tau}: eta0 comparison, steps {start_step}-{end_step} "
        f"({title_suffix}, trailing {rolling_window}-step window)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, output_base)


def _plot_fixed_tau(
    output_base: Path,
    grouped: dict[tuple[int, float], list[dict[str, float | int | str]]],
    tau: int,
    start_step: int,
    end_step: int,
    rolling_window: int,
) -> list[Path]:
    panels = [
        ("e0", "ground energy E0 (Ha)", 0.002),
        ("e1", "excited energy E1 (Ha)", 0.002),
        ("gap_ev", "gap E1 - E0 (eV)", 0.04),
        ("spin", "mean <S^2>", 0.0002),
    ]
    return _plot_fixed_tau_panels(
        output_base,
        grouped,
        tau,
        start_step,
        end_step,
        rolling_window,
        panels,
        "mean",
        "mean",
    )


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
    for key in ("energy", "ewmean", "ewvar", "e0", "e1", "gap_hartree", "gap_ev", "spin", "spin_state_0", "spin_state_1", "pmove"):
        values = _series(rows, key)
        delta = np.full(values.shape, np.nan, dtype=float)
        finite_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        delta[1:][finite_pairs] = np.abs(values[1:][finite_pairs] - values[:-1][finite_pairs])
        first_mean = float(np.nanmean(values[first_after_mask]))
        last_mean = float(np.nanmean(values[last_mask]))
        out[f"after{start_step}_{key}_mean"] = float(np.nanmean(values[after_mask]))
        out[f"after{start_step}_{key}_std"] = float(np.nanstd(values[after_mask], ddof=1))
        out[f"after{start_step}_{key}_var"] = float(np.nanvar(values[after_mask], ddof=1))
        out[f"first_after{start_step}_{rolling_window}_{key}_mean"] = first_mean
        out[f"first_after{start_step}_{rolling_window}_{key}_std"] = float(np.nanstd(values[first_after_mask], ddof=1))
        out[f"first_after{start_step}_{rolling_window}_{key}_var"] = float(np.nanvar(values[first_after_mask], ddof=1))
        out[f"last{rolling_window}_{key}_mean"] = last_mean
        out[f"last{rolling_window}_{key}_std"] = float(np.nanstd(values[last_mask], ddof=1))
        out[f"last{rolling_window}_{key}_var"] = float(np.nanvar(values[last_mask], ddof=1))
        out[f"last_vs_first_after{start_step}_{rolling_window}_{key}_delta"] = last_mean - first_mean
        out[f"last{rolling_window}_{key}_abs_step_delta_mean"] = float(np.nanmean(delta[last_mask]))
        out[f"last{rolling_window}_{key}_abs_step_delta_std"] = float(np.nanstd(delta[last_mask], ddof=1))
    return out


def _write_summary_csv(path: Path, summaries: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def _write_markdown(
    path: Path,
    summaries: list[dict[str, float | int | str]],
    plot_paths: list[Path],
    rolling_window: int,
    start_step: int,
    title: str,
) -> None:
    def fmt(value: object, digits: int = 6) -> str:
        return f"{float(value):.{digits}f}"

    lines = [
        f"# {title}",
        "",
        f"Data combine tasks 0113 and 0114. Mean plots use trailing {rolling_window}-step means after step {start_step}.",
        f"Fluctuation plots use the same trailing {rolling_window}-step window for variance/std/absolute one-step changes.",
        "",
        f"Final-window statistics use steps {30000 - rolling_window}-{29999}.",
        "",
        "| Tau | eta0 | Rows | E0 mean Ha | E0 std | E1 mean Ha | E1 std | Gap eV | Gap std | Spin mean | Spin std | EWVar mean | EWVar std |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda r: (int(r["tau"]), float(r["eta0"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tau"]),
                    f"{float(row['eta0']):g}",
                    str(row["num_rows"]),
                    fmt(row[f"last{rolling_window}_e0_mean"]),
                    fmt(row[f"last{rolling_window}_e0_std"]),
                    fmt(row[f"last{rolling_window}_e1_mean"]),
                    fmt(row[f"last{rolling_window}_e1_std"]),
                    fmt(row[f"last{rolling_window}_gap_ev_mean"], 4),
                    fmt(row[f"last{rolling_window}_gap_ev_std"], 4),
                    fmt(row[f"last{rolling_window}_spin_mean"]),
                    fmt(row[f"last{rolling_window}_spin_std"]),
                    fmt(row[f"last{rolling_window}_ewvar_mean"]),
                    fmt(row[f"last{rolling_window}_ewvar_std"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Drift And Step Changes",
            "",
            f"Delta columns compare the first {rolling_window}-step window after {start_step} against the final {rolling_window}-step window.",
            "",
            "| Tau | eta0 | dE0 Ha | dE1 Ha | dGap eV | dSpin | mean abs dE0/step | mean abs dE1/step | mean abs dGap/step | mean abs dSpin/step |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summaries, key=lambda r: (int(r["tau"]), float(r["eta0"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tau"]),
                    f"{float(row['eta0']):g}",
                    fmt(row[f"last_vs_first_after{start_step}_{rolling_window}_e0_delta"]),
                    fmt(row[f"last_vs_first_after{start_step}_{rolling_window}_e1_delta"]),
                    fmt(row[f"last_vs_first_after{start_step}_{rolling_window}_gap_ev_delta"], 4),
                    fmt(row[f"last_vs_first_after{start_step}_{rolling_window}_spin_delta"]),
                    fmt(row[f"last{rolling_window}_e0_abs_step_delta_mean"]),
                    fmt(row[f"last{rolling_window}_e1_abs_step_delta_mean"]),
                    fmt(row[f"last{rolling_window}_gap_ev_abs_step_delta_mean"], 4),
                    fmt(row[f"last{rolling_window}_spin_abs_step_delta_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Generated Plots", ""])
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path.resolve()}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison"),
    )
    parser.add_argument("--rolling-window", type=int, default=1000)
    parser.add_argument("--end-step", type=int, default=29999)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, float], list[dict[str, float | int | str]]] = {}
    all_rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    for spec in _default_run_specs(repo_root):
        rows, spec_warnings = _rows_for_spec(spec)
        grouped[(spec.tau, spec.eta0)] = rows
        all_rows.extend(rows)
        warnings.extend(spec_warnings)
        steps = _steps(rows)
        if len(rows) != 30000 or int(steps[0]) != 0 or int(steps[-1]) != 29999:
            warnings.append(f"{spec.run}: rows={len(rows)}, steps={int(steps[0])}-{int(steps[-1])}")

    prefix = "0113_0114_fixed_tau_eta_comparison"
    _write_timeseries(output_dir / f"{prefix}_combined_timeseries.csv", all_rows)

    all_plot_paths: list[Path] = []
    summaries_by_window: dict[str, list[dict[str, float | int | str]]] = {}
    for window_name, start_step in (("after5000", 5000), ("last10000", 20000)):
        summaries = [
            _summary_for_rows(grouped[(tau, eta0)], args.rolling_window, start_step)
            for tau in (10000, 15000, 20000)
            for eta0 in (0.001, 0.005, 0.01, 0.02)
        ]
        summaries_by_window[window_name] = summaries
        _write_summary_csv(output_dir / f"{prefix}_{window_name}_summary.csv", summaries)
        (output_dir / f"{prefix}_{window_name}_summary.json").write_text(
            json.dumps({"summary": summaries, "warnings": warnings}, indent=2, sort_keys=True)
        )

        plot_paths: list[Path] = []
        for tau in (10000, 15000, 20000):
            output_base = output_dir / f"{prefix}_tau{tau}_rolling_after{start_step}_window{args.rolling_window}"
            plot_paths.extend(
                _plot_fixed_tau(
                    output_base,
                    grouped,
                    tau,
                    start_step,
                    args.end_step,
                    args.rolling_window,
                )
            )
            fluctuation_panels = [
                ("e0", "rolling std E0 (Ha)", 0.0002),
                ("e1", "rolling std E1 (Ha)", 0.0002),
                ("gap_ev", "rolling std gap (eV)", 0.004),
                ("spin", "rolling std <S^2>", 0.00002),
            ]
            output_base = output_dir / f"{prefix}_tau{tau}_rolling_std_after{start_step}_window{args.rolling_window}"
            plot_paths.extend(
                _plot_fixed_tau_panels(
                    output_base,
                    grouped,
                    tau,
                    start_step,
                    args.end_step,
                    args.rolling_window,
                    fluctuation_panels,
                    "std",
                    "standard deviation",
                )
            )
            variance_panels = [
                ("e0", "rolling var E0 (Ha^2)", 0.00001),
                ("e1", "rolling var E1 (Ha^2)", 0.00001),
                ("gap_ev", "rolling var gap (eV^2)", 0.0001),
                ("spin", "rolling var <S^2>", 0.00000001),
            ]
            output_base = output_dir / f"{prefix}_tau{tau}_rolling_var_after{start_step}_window{args.rolling_window}"
            plot_paths.extend(
                _plot_fixed_tau_panels(
                    output_base,
                    grouped,
                    tau,
                    start_step,
                    args.end_step,
                    args.rolling_window,
                    variance_panels,
                    "var",
                    "variance",
                )
            )
            delta_panels = [
                ("e0", "rolling mean |dE0|/step (Ha)", 0.00002),
                ("e1", "rolling mean |dE1|/step (Ha)", 0.00002),
                ("gap_ev", "rolling mean |dGap|/step (eV)", 0.0005),
                ("spin", "rolling mean |d<S^2>|/step", 0.000002),
            ]
            output_base = output_dir / f"{prefix}_tau{tau}_rolling_abs_delta_after{start_step}_window{args.rolling_window}"
            plot_paths.extend(
                _plot_fixed_tau_panels(
                    output_base,
                    grouped,
                    tau,
                    start_step,
                    args.end_step,
                    args.rolling_window,
                    delta_panels,
                    "abs_delta_mean",
                    "mean absolute one-step change",
                )
            )
            ewvar_panels = [("ewvar", "EW variance (Ha^2)", 0.00001)]
            output_base = output_dir / f"{prefix}_tau{tau}_ewvar_rolling_after{start_step}_window{args.rolling_window}"
            plot_paths.extend(
                _plot_fixed_tau_panels(
                    output_base,
                    grouped,
                    tau,
                    start_step,
                    args.end_step,
                    args.rolling_window,
                    ewvar_panels,
                    "mean",
                    "mean EW variance",
                )
            )
        all_plot_paths.extend(plot_paths)
        _write_markdown(
            output_dir / f"{prefix}_{window_name}_comparison.md",
            summaries,
            plot_paths,
            args.rolling_window,
            start_step,
            f"0113/0114 Fixed-Tau Eta0 Comparison ({window_name})",
        )

    (output_dir / f"{prefix}_warnings.json").write_text(json.dumps(warnings, indent=2))
    print(f"warnings: {warnings}")
    print(f"Wrote {output_dir / f'{prefix}_combined_timeseries.csv'}")
    for window_name in summaries_by_window:
        print(f"Wrote {output_dir / f'{prefix}_{window_name}_comparison.md'}")
    for plot_path in all_plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
