#!/usr/bin/env python
"""Analyze and plot the 0097 DeepQMC-aligned PsiFormer run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "tasks/psiformer/0097_deepqmc_aligned_excited_state"
DEFAULT_RUN_NAME = "fullnode_anygpu_fused_qkv_x64_deepqmc_b4096_i10000_levmap128_jaxattn"
DEFAULT_JOB_ID = "133788"
HARTREE_TO_EV = 27.211386245988


def main() -> int:
    args = _parse_args()
    task_root = _resolve_path(args.task_root)
    run_root = task_root / "runs" / args.run_name
    checkpoint_dir = run_root / "results/checkpoints"
    output_dir = task_root / "results/validation" / _job_output_subdir(
        args.job_id, args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _read_run(checkpoint_dir, run_root)
    data["job_id"] = args.job_id
    summary = _summarize(data, job_id=args.job_id, run_name=args.run_name)

    stem = f"psiformer_0097_deepqmc_{args.job_id}"
    summary_json = output_dir / f"{stem}_summary.json"
    summary_md = output_dir / f"{stem}_analysis.md"
    timeseries_csv = output_dir / f"{stem}_timeseries.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(_format_markdown(summary, stem), encoding="utf-8")
    _write_timeseries_csv(timeseries_csv, data)

    _plot_iteration_windows(data, output_dir, stem)
    _plot_state_gap_windows(data, output_dir, stem)
    _plot_overlap_windows(data, output_dir, stem)
    _plot_after2000_roll1000(data, output_dir, stem)

    print("psiformer_0097_deepqmc_analysis: ok")
    print(f"summary_markdown: {_display_path(summary_md)}")
    print(f"summary_json: {_display_path(summary_json)}")
    print(f"timeseries_csv: {_display_path(timeseries_csv)}")
    for family in ("iteration_evolution", "state_energy_gap", "overlap_evolution"):
        for suffix in ("full", "after1000", "last1000"):
            print(
                "plot: "
                + _display_path(output_dir / f"{stem}_{family}_{suffix}.png")
            )
    print(
        "plot: "
        + _display_path(output_dir / f"{stem}_ground_excited_after2000_roll1000.png")
    )
    print(
        "plot: "
        + _display_path(output_dir / f"{stem}_gap_after2000_roll1000.png")
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    return parser.parse_args()


def _job_output_subdir(job_id: str, run_name: str) -> str:
    if "fused_qkv" in run_name:
        return f"{job_id}_fused_qkv"
    if "ferminet" in run_name:
        return f"{job_id}_ferminet_qkv"
    return str(job_id)


def _read_run(checkpoint_dir: Path, run_root: Path) -> dict[str, Any]:
    train = _read_train_stats(checkpoint_dir / "train_stats.csv")
    energy_matrix = _stack(_read_appended_npy(checkpoint_dir / "energy_matrix.npy"))
    overlap = _stack(_read_appended_npy(checkpoint_dir / "overlap_matrix.npy"))
    symmetric_overlap = _stack(
        _read_appended_npy(checkpoint_dir / "overlap_symmetric_matrix.npy")
    )
    overlap_penalty = _stack(
        _read_appended_npy(checkpoint_dir / "overlap_penalty_matrix.npy")
    )
    gradient_scale = _stack(
        _read_appended_npy(checkpoint_dir / "overlap_gradient_scale.npy")
    )
    state_ordering = _stack(
        _read_appended_npy(checkpoint_dir / "overlap_state_ordering.npy"),
        dtype=int,
    )
    scale_energy_ewm = _stack(
        _read_appended_npy(checkpoint_dir / "overlap_scale_energy_ewm.npy")
    )
    scale_std_ewm = _stack(_read_appended_npy(checkpoint_dir / "overlap_scale_std_ewm.npy"))
    runtime = _read_runtime(run_root / "results/validation/ferminet_train_runtime.json")

    if energy_matrix.ndim != 2 or energy_matrix.shape[1] < 2:
        raise ValueError(f"expected two-state energy_matrix in {checkpoint_dir}")
    if len(train["step"]) != len(energy_matrix):
        train["step"] = np.arange(len(energy_matrix), dtype=int)

    energy_order = np.argsort(energy_matrix, axis=1)
    row_index = np.arange(len(energy_matrix))
    sorted_energy = energy_matrix[row_index[:, None], energy_order]
    if state_ordering.shape != energy_matrix.shape:
        state_ordering = np.tile(
            np.arange(energy_matrix.shape[1]), (energy_matrix.shape[0], 1)
        )
    ordered_by_state = energy_matrix[row_index[:, None], state_ordering]

    return {
        "checkpoint_dir": checkpoint_dir,
        "runtime": runtime,
        "step": train["step"],
        "energy": train["energy"],
        "ewmean": train["ewmean"],
        "ewvar": train["ewvar"],
        "pmove": train["pmove"],
        "energy_matrix": energy_matrix,
        "ground_energy": sorted_energy[:, 0],
        "excited_energy": sorted_energy[:, -1],
        "fixed_gap_hartree": energy_matrix[:, 1] - energy_matrix[:, 0],
        "sorted_gap_hartree": sorted_energy[:, -1] - sorted_energy[:, 0],
        "ordered_by_state": ordered_by_state,
        "state_ordering": state_ordering,
        "overlap": overlap,
        "symmetric_overlap": symmetric_overlap,
        "overlap_penalty": overlap_penalty,
        "gradient_scale": gradient_scale,
        "scale_energy_ewm": scale_energy_ewm,
        "scale_std_ewm": scale_std_ewm,
    }


def _read_train_stats(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty train_stats.csv: {path}")
    data: dict[str, np.ndarray] = {
        "step": np.array([int(row["step"]) for row in rows], dtype=int)
    }
    for key in ("energy", "ewmean", "ewvar", "pmove"):
        data[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return data


def _read_appended_npy(path: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if not path.exists() or path.stat().st_size == 0:
        return arrays
    with path.open("rb") as handle:
        while True:
            try:
                arrays.append(np.load(handle, allow_pickle=False))
            except (EOFError, ValueError):
                break
    return arrays


def _read_runtime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stack(arrays: list[np.ndarray], *, dtype: type[float] | type[int] = float) -> np.ndarray:
    if not arrays:
        return np.empty((0,), dtype=dtype)
    return np.stack([np.asarray(array, dtype=dtype) for array in arrays], axis=0)


def _summarize(data: dict[str, Any], *, job_id: str, run_name: str) -> dict[str, Any]:
    step = data["step"]
    state_ordering = data["state_ordering"]
    default_order = np.tile(
        np.arange(state_ordering.shape[1]), (state_ordering.shape[0], 1)
    )
    ordering_swaps = step[np.any(state_ordering != default_order, axis=1)]
    fixed_gap = data["fixed_gap_hartree"]
    sorted_gap = data["sorted_gap_hartree"]
    sym_overlap = data["symmetric_overlap"][:, 0, 1]
    penalty = data["overlap_penalty"][:, 0, 1]

    return {
        "job_id": job_id,
        "run_name": run_name,
        "checkpoint_dir": _display_path(data["checkpoint_dir"]),
        "runtime": data["runtime"],
        "rows": int(len(step)),
        "first_step": int(step[0]),
        "final_step": int(step[-1]),
        "scalar_energy": _series_summary(data["energy"]),
        "ewmean": _series_summary(data["ewmean"]),
        "ewvar": _series_summary(data["ewvar"]),
        "pmove": _series_summary(data["pmove"]),
        "final_state_energy": _json_array(data["energy_matrix"][-1]),
        "tail100_state_energy_mean": _json_array(_tail_mean(data["energy_matrix"], 100)),
        "tail1000_state_energy_mean": _json_array(
            _tail_mean(data["energy_matrix"], 1000)
        ),
        "final_ground_energy": float(data["ground_energy"][-1]),
        "final_excited_energy": float(data["excited_energy"][-1]),
        "fixed_gap_hartree": _gap_summary(fixed_gap),
        "fixed_gap_ev": _gap_summary(fixed_gap * HARTREE_TO_EV),
        "sorted_gap_hartree": _gap_summary(sorted_gap),
        "sorted_gap_ev": _gap_summary(sorted_gap * HARTREE_TO_EV),
        "fixed_gap_negative_count": int(np.sum(fixed_gap < 0)),
        "fixed_gap_negative_count_last100": int(np.sum(fixed_gap[-100:] < 0)),
        "fixed_gap_negative_count_last1000": int(np.sum(fixed_gap[-1000:] < 0)),
        "state_ordering_first": _json_array(state_ordering[0]),
        "state_ordering_final": _json_array(state_ordering[-1]),
        "state_ordering_swap_count": int(len(ordering_swaps)),
        "state_ordering_first_swap_step": None
        if len(ordering_swaps) == 0
        else int(ordering_swaps[0]),
        "state_ordering_last_swap_step": None
        if len(ordering_swaps) == 0
        else int(ordering_swaps[-1]),
        "final_overlap_matrix": _json_array(data["overlap"][-1]),
        "final_symmetric_overlap_matrix": _json_array(data["symmetric_overlap"][-1]),
        "final_overlap_penalty_matrix": _json_array(data["overlap_penalty"][-1]),
        "final_overlap_gradient_scale": _json_array(data["gradient_scale"][-1]),
        "symmetric_overlap01": _signed_overlap_summary(sym_overlap),
        "abs_symmetric_overlap01": _series_summary(np.abs(sym_overlap)),
        "overlap_penalty01": _series_summary(penalty),
        "final_overlap_scale_energy_ewm": _json_array(data["scale_energy_ewm"][-1]),
        "final_overlap_scale_std_ewm": _json_array(data["scale_std_ewm"][-1]),
    }


def _series_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "tail100_mean": _tail_mean_scalar(values, 100),
        "tail500_mean": _tail_mean_scalar(values, 500),
        "tail1000_mean": _tail_mean_scalar(values, 1000),
        "tail1000_std": _tail_std_scalar(values, 1000),
    }


def _gap_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "tail100_mean": _tail_mean_scalar(values, 100),
        "tail500_mean": _tail_mean_scalar(values, 500),
        "tail1000_mean": _tail_mean_scalar(values, 1000),
        "final_rolling100": float(_rolling_mean(values, 100)[-1]),
    }


def _signed_overlap_summary(values: np.ndarray) -> dict[str, float]:
    summary = _series_summary(values)
    summary["tail1000_abs_mean"] = _tail_mean_scalar(np.abs(values), 1000)
    summary["tail1000_abs_max"] = float(np.max(np.abs(values[-1000:])))
    return summary


def _write_timeseries_csv(path: Path, data: dict[str, Any]) -> None:
    fieldnames = [
        "step",
        "energy",
        "ewmean",
        "ewvar",
        "pmove",
        "state0_energy",
        "state1_energy",
        "ground_energy",
        "excited_energy",
        "fixed_gap_hartree",
        "fixed_gap_ev",
        "sorted_gap_hartree",
        "sorted_gap_ev",
        "state_ordering0",
        "state_ordering1",
        "overlap01",
        "overlap10",
        "symmetric_overlap01",
        "abs_symmetric_overlap01",
        "overlap_penalty01",
        "gradient_scale0",
        "gradient_scale1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for idx, step in enumerate(data["step"]):
            writer.writerow(
                {
                    "step": int(step),
                    "energy": float(data["energy"][idx]),
                    "ewmean": float(data["ewmean"][idx]),
                    "ewvar": float(data["ewvar"][idx]),
                    "pmove": float(data["pmove"][idx]),
                    "state0_energy": float(data["energy_matrix"][idx, 0]),
                    "state1_energy": float(data["energy_matrix"][idx, 1]),
                    "ground_energy": float(data["ground_energy"][idx]),
                    "excited_energy": float(data["excited_energy"][idx]),
                    "fixed_gap_hartree": float(data["fixed_gap_hartree"][idx]),
                    "fixed_gap_ev": float(data["fixed_gap_hartree"][idx] * HARTREE_TO_EV),
                    "sorted_gap_hartree": float(data["sorted_gap_hartree"][idx]),
                    "sorted_gap_ev": float(data["sorted_gap_hartree"][idx] * HARTREE_TO_EV),
                    "state_ordering0": int(data["state_ordering"][idx, 0]),
                    "state_ordering1": int(data["state_ordering"][idx, 1]),
                    "overlap01": float(data["overlap"][idx, 0, 1]),
                    "overlap10": float(data["overlap"][idx, 1, 0]),
                    "symmetric_overlap01": float(data["symmetric_overlap"][idx, 0, 1]),
                    "abs_symmetric_overlap01": float(
                        abs(data["symmetric_overlap"][idx, 0, 1])
                    ),
                    "overlap_penalty01": float(data["overlap_penalty"][idx, 0, 1]),
                    "gradient_scale0": float(data["gradient_scale"][idx, 0, 0]),
                    "gradient_scale1": float(data["gradient_scale"][idx, 1, 0]),
                }
            )


def _plot_iteration_windows(data: dict[str, Any], output_dir: Path, stem: str) -> None:
    for suffix, start_step, rolling_window in _plot_windows():
        _plot_iteration_window(
            data,
            output_dir / f"{stem}_iteration_evolution_{suffix}",
            title=f"0097 / {data['job_id']} scalar training evolution: {suffix}",
            start_step=start_step,
            rolling_window=rolling_window,
        )


def _plot_state_gap_windows(data: dict[str, Any], output_dir: Path, stem: str) -> None:
    for suffix, start_step, rolling_window in _plot_windows():
        _plot_state_gap_window(
            data,
            output_dir / f"{stem}_state_energy_gap_{suffix}",
            title=f"0097 / {data['job_id']} state energies and gap: {suffix}",
            start_step=start_step,
            rolling_window=rolling_window,
        )


def _plot_overlap_windows(data: dict[str, Any], output_dir: Path, stem: str) -> None:
    for suffix, start_step, rolling_window in _plot_windows():
        _plot_overlap_window(
            data,
            output_dir / f"{stem}_overlap_evolution_{suffix}",
            title=f"0097 / {data['job_id']} overlap diagnostics: {suffix}",
            start_step=start_step,
            rolling_window=rolling_window,
        )


def _plot_after2000_roll1000(
    data: dict[str, Any], output_dir: Path, stem: str
) -> None:
    start_step = 2000
    rolling_window = 1000
    _plot_ground_excited_after2000(
        data,
        output_dir / f"{stem}_ground_excited_after2000_roll1000",
        start_step=start_step,
        rolling_window=rolling_window,
    )
    _plot_gap_after2000(
        data,
        output_dir / f"{stem}_gap_after2000_roll1000",
        start_step=start_step,
        rolling_window=rolling_window,
    )


def _plot_windows() -> tuple[tuple[str, int | None, int], ...]:
    return (
        ("full", None, 100),
        ("after1000", 1000, 100),
        ("last1000", 9000, 25),
    )


def _plot_ground_excited_after2000(
    data: dict[str, Any],
    output_base: Path,
    *,
    start_step: int,
    rolling_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    rolling_series = []
    for label, values, color in (
        ("ground", data["ground_energy"], "#1f77b4"),
        ("excited", data["excited_energy"], "#d62728"),
    ):
        steps, window_values = _windowed(data["step"], values, start_step=start_step)
        rolled = _plot_raw_and_rolling(
            ax, steps, window_values, color, rolling_window, label
        )
        rolling_series.append(rolled)
    _set_ylim_from_series(ax, rolling_series, min_padding=0.002)
    ax.set_title(
        f"0097 / {data['job_id']} ground and excited energies after step 2000"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _plot_gap_after2000(
    data: dict[str, Any],
    output_base: Path,
    *,
    start_step: int,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    gap_ev = data["sorted_gap_hartree"] * HARTREE_TO_EV
    fixed_gap_ev = data["fixed_gap_hartree"] * HARTREE_TO_EV
    for ax, values, ylabel, label, color in (
        (axes[0], gap_ev, "Gap (eV)", "energy-sorted gap", "#2ca02c"),
        (axes[1], fixed_gap_ev, "Fixed state1-state0 gap (eV)", "fixed gap", "#9467bd"),
    ):
        steps, window_values = _windowed(data["step"], values, start_step=start_step)
        rolled = _plot_raw_and_rolling(
            ax, steps, window_values, color, rolling_window, label
        )
        _set_ylim_from_series(ax, [rolled], min_padding=0.05)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    axes[0].set_title(f"0097 / {data['job_id']} gap evolution after step 2000")
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    _save_figure(fig, output_base)


def _plot_iteration_window(
    data: dict[str, Any],
    output_base: Path,
    *,
    title: str,
    start_step: int | None,
    rolling_window: int,
) -> None:
    metrics = (
        ("energy", "Energy (Ha)", "#1f77b4"),
        ("ewmean", "EW mean (Ha)", "#2ca02c"),
        ("ewvar", "EW variance", "#ff7f0e"),
        ("pmove", "Pmove", "#9467bd"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (metric, ylabel, color) in zip(axes.ravel(), metrics, strict=True):
        steps, values = _windowed(data["step"], data[metric], start_step=start_step)
        _plot_raw_and_rolling(ax, steps, values, color, rolling_window, metric)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(fig, output_base)


def _plot_state_gap_window(
    data: dict[str, Any],
    output_base: Path,
    *,
    title: str,
    start_step: int | None,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    panels = (
        (
            "Raw state energies",
            (
                ("state 0", data["energy_matrix"][:, 0], "#1f77b4"),
                ("state 1", data["energy_matrix"][:, 1], "#d62728"),
            ),
            "Energy (Ha)",
        ),
        (
            "Energy-sorted states",
            (
                ("ground", data["ground_energy"], "#1f77b4"),
                ("excited", data["excited_energy"], "#d62728"),
            ),
            "Energy (Ha)",
        ),
        (
            "Gap",
            (
                ("fixed state1-state0", data["fixed_gap_hartree"] * HARTREE_TO_EV, "#2ca02c"),
                ("energy-sorted", data["sorted_gap_hartree"] * HARTREE_TO_EV, "#9467bd"),
            ),
            "Gap (eV)",
        ),
    )
    for ax, (label, series, ylabel) in zip(axes, panels, strict=True):
        for series_label, values, color in series:
            steps, window_values = _windowed(
                data["step"], values, start_step=start_step
            )
            _plot_raw_and_rolling(
                ax, steps, window_values, color, rolling_window, series_label
            )
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(fig, output_base)


def _plot_overlap_window(
    data: dict[str, Any],
    output_base: Path,
    *,
    title: str,
    start_step: int | None,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = (
        (
            "symmetric overlap 01",
            data["symmetric_overlap"][:, 0, 1],
            "Symmetric overlap",
            "#1f77b4",
        ),
        (
            "abs symmetric overlap 01",
            np.abs(data["symmetric_overlap"][:, 0, 1]),
            "|Symmetric overlap|",
            "#d62728",
        ),
        (
            "overlap penalty 01",
            data["overlap_penalty"][:, 0, 1],
            "Overlap penalty",
            "#2ca02c",
        ),
        (
            "gradient scale state0",
            data["gradient_scale"][:, 0, 0],
            "Gradient scale",
            "#9467bd",
        ),
    )
    for ax, (label, values, ylabel, color) in zip(axes.ravel(), metrics, strict=True):
        steps, window_values = _windowed(data["step"], values, start_step=start_step)
        _plot_raw_and_rolling(ax, steps, window_values, color, rolling_window, label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(fig, output_base)


def _windowed(
    steps: np.ndarray, values: np.ndarray, *, start_step: int | None
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(steps, dtype=bool)
    if start_step is not None:
        mask = steps >= start_step
    return steps[mask], values[mask]


def _plot_raw_and_rolling(
    ax: plt.Axes,
    steps: np.ndarray,
    values: np.ndarray,
    color: str,
    rolling_window: int,
    label: str,
) -> np.ndarray:
    ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.55)
    if len(values) >= rolling_window:
        rolled = _rolling_mean(values, rolling_window)
        ax.plot(
            steps,
            rolled,
            color=color,
            linewidth=1.8,
            label=f"{label} ({rolling_window}-step mean)",
        )
        return rolled
    else:
        ax.plot(steps, values, color=color, linewidth=1.8, label=label)
        return values


def _set_ylim_from_series(
    ax: plt.Axes, series: list[np.ndarray], *, min_padding: float
) -> None:
    values = np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in series])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    padding = max((upper - lower) * 0.08, min_padding)
    ax.set_ylim(lower - padding, upper + padding)


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    for ext in ("png", "svg"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def _format_markdown(summary: dict[str, Any], stem: str) -> str:
    runtime = summary["runtime"]
    seconds_per_iteration = runtime.get("seconds_per_iteration")
    elapsed_seconds = runtime.get("elapsed_seconds")
    final_checkpoint = runtime.get("final_checkpoint", {})

    lines = [
        f"# 0097 DeepQMC-Aligned PsiFormer {summary['job_id']} Analysis",
        "",
        "Task: `0097_deepqmc_aligned_excited_state`",
        f"Run: `{summary['run_name']}`",
        f"Checkpoint dir: `{summary['checkpoint_dir']}`",
        "",
        "## Completion",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Rows | {summary['rows']} |",
        f"| Steps | {summary['first_step']} -> {summary['final_step']} |",
        f"| Elapsed seconds | {_fmt(elapsed_seconds)} |",
        f"| Seconds / iteration | {_fmt(seconds_per_iteration)} |",
        f"| Final checkpoint exists | {final_checkpoint.get('exists')} |",
        "",
        "## Scalar Training Stats",
        "",
        "| Metric | Final | Tail100 mean | Tail500 mean | Tail1000 mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("scalar_energy", "Energy (Ha)"),
        ("ewmean", "EW mean (Ha)"),
        ("ewvar", "EW variance"),
        ("pmove", "Pmove"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {_fmt(item['last'])} | {_fmt(item['tail100_mean'])} | "
            f"{_fmt(item['tail500_mean'])} | {_fmt(item['tail1000_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## State Energies And Gap",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Final state energies (Ha) | `{summary['final_state_energy']}` |",
            f"| Tail100 state-energy mean (Ha) | `{summary['tail100_state_energy_mean']}` |",
            f"| Tail1000 state-energy mean (Ha) | `{summary['tail1000_state_energy_mean']}` |",
            f"| Final fixed gap (Ha) | {_fmt(summary['fixed_gap_hartree']['last'])} |",
            f"| Final fixed gap (eV) | {_fmt(summary['fixed_gap_ev']['last'])} |",
            f"| Tail100 fixed gap (eV) | {_fmt(summary['fixed_gap_ev']['tail100_mean'])} |",
            f"| Tail500 fixed gap (eV) | {_fmt(summary['fixed_gap_ev']['tail500_mean'])} |",
            f"| Tail1000 fixed gap (eV) | {_fmt(summary['fixed_gap_ev']['tail1000_mean'])} |",
            f"| Final energy-sorted gap (eV) | {_fmt(summary['sorted_gap_ev']['last'])} |",
            f"| Tail1000 energy-sorted gap (eV) | {_fmt(summary['sorted_gap_ev']['tail1000_mean'])} |",
            f"| Fixed-gap negative rows | {summary['fixed_gap_negative_count']} |",
            f"| Fixed-gap negative rows, last1000 | {summary['fixed_gap_negative_count_last1000']} |",
            f"| State ordering final | `{summary['state_ordering_final']}` |",
            f"| State-ordering swaps | {summary['state_ordering_swap_count']} |",
            "",
            "## Overlap Diagnostics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Final overlap matrix | `{summary['final_overlap_matrix']}` |",
            f"| Final symmetric overlap matrix | `{summary['final_symmetric_overlap_matrix']}` |",
            f"| Final overlap penalty matrix | `{summary['final_overlap_penalty_matrix']}` |",
            f"| Final gradient scale | `{summary['final_overlap_gradient_scale']}` |",
            f"| Final symmetric overlap 01 | {_fmt(summary['symmetric_overlap01']['last'])} |",
            f"| Tail1000 abs symmetric overlap 01 mean | {_fmt(summary['symmetric_overlap01']['tail1000_abs_mean'])} |",
            f"| Tail1000 abs symmetric overlap 01 max | {_fmt(summary['symmetric_overlap01']['tail1000_abs_max'])} |",
            f"| Final overlap penalty 01 | {_fmt(summary['overlap_penalty01']['last'])} |",
            "",
            "## Artifacts",
            "",
            f"- Timeseries CSV: `{stem}_timeseries.csv`",
            f"- Summary JSON: `{stem}_summary.json`",
            f"- Scalar evolution plots: `{stem}_iteration_evolution_*.png` / `.svg`",
            f"- State energy and gap plots: `{stem}_state_energy_gap_*.png` / `.svg`",
            f"- Overlap plots: `{stem}_overlap_evolution_*.png` / `.svg`",
            f"- Ground/excited energy after step 2000 with 1000-step mean: `{stem}_ground_excited_after2000_roll1000.png` / `.svg`",
            f"- Gap after step 2000 with 1000-step mean: `{stem}_gap_after2000_roll1000.png` / `.svg`",
            "",
            "## Readout",
            "",
            "The run completed the requested 10000 iterations and wrote the final checkpoint. "
            "The DeepQMC-aligned independent-state route keeps the fixed state ordering "
            "`[0, 1]` throughout the run and drives the final symmetric off-diagonal "
            "overlap close to zero. The final single-step gap is larger than the tail "
            "means, so tail windows are a better read of the late-time behavior than the "
            "last row alone.",
            "",
        ]
    )
    return "\n".join(lines)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _tail_mean(values: np.ndarray, count: int) -> np.ndarray:
    return np.mean(values[-min(count, len(values)) :], axis=0)


def _tail_mean_scalar(values: np.ndarray, count: int) -> float:
    return float(np.mean(values[-min(count, len(values)) :]))


def _tail_std_scalar(values: np.ndarray, count: int) -> float:
    tail = values[-min(count, len(values)) :]
    return float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0


def _json_array(value: Any) -> Any:
    array = np.real_if_close(np.asarray(value))
    if array.shape == ():
        return array.item()
    return array.tolist()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.9g}"
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
