#!/usr/bin/env python
"""Plot retry-1 0097 DeepQMC-aligned PsiFormer method comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "tasks/psiformer/0097_deepqmc_aligned_excited_state"
DEFAULT_SERIES = {
    "fused-QKV / 133788": "133788_fused_qkv/psiformer_0097_deepqmc_133788_timeseries.csv",
    "FermiNet Q/K/V / 133789": "133789_ferminet_qkv/psiformer_0097_deepqmc_133789_timeseries.csv",
}


def main() -> int:
    args = _parse_args()
    task_root = _resolve_path(args.task_root)
    validation_dir = task_root / "results/validation"
    output_dir = validation_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    series = {
        label: _read_timeseries(validation_dir / filename)
        for label, filename in DEFAULT_SERIES.items()
    }

    energy_path = output_dir / "psiformer_0097_deepqmc_retry1_ground_excited_after2000_roll1000_comparison"
    gap_path = output_dir / "psiformer_0097_deepqmc_retry1_gap_after2000_roll1000_comparison"
    _plot_ground_excited(series, energy_path, args.start_step, args.rolling_window)
    _plot_gap(series, gap_path, args.start_step, args.rolling_window)

    print("psiformer_0097_deepqmc_retry1_comparison: ok")
    print(f"plot: {_display_path(energy_path.with_suffix('.png'))}")
    print(f"plot: {_display_path(gap_path.with_suffix('.png'))}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--start-step", type=int, default=2000)
    parser.add_argument("--rolling-window", type=int, default=1000)
    return parser.parse_args()


def _read_timeseries(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty timeseries: {path}")
    return {
        "step": np.array([int(row["step"]) for row in rows], dtype=int),
        "ground_energy": np.array(
            [float(row["ground_energy"]) for row in rows], dtype=float
        ),
        "excited_energy": np.array(
            [float(row["excited_energy"]) for row in rows], dtype=float
        ),
        "gap_ev": np.array([float(row["sorted_gap_ev"]) for row in rows], dtype=float),
    }


def _plot_ground_excited(
    series: dict[str, dict[str, np.ndarray]],
    output_base: Path,
    start_step: int,
    rolling_window: int,
) -> None:
    colors = {
        "fused-QKV / 133788": "#d62728",
        "FermiNet Q/K/V / 133789": "#1f77b4",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    panels = (
        ("ground_energy", "Ground-state energy (Ha)", "Ground state"),
        ("excited_energy", "Excited-state energy (Ha)", "Excited state"),
    )
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        rolled_values = []
        for label, data in series.items():
            steps, values = _windowed(
                data["step"], data[metric], start_step=start_step
            )
            rolled = _plot_raw_and_rolling(
                ax, steps, values, colors[label], rolling_window, label
            )
            rolled_values.append(rolled)
        _set_ylim_from_series(ax, rolled_values, min_padding=0.001)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(
        f"0097 retry-1 ground/excited energies after step {start_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_base)


def _plot_gap(
    series: dict[str, dict[str, np.ndarray]],
    output_base: Path,
    start_step: int,
    rolling_window: int,
) -> None:
    colors = {
        "fused-QKV / 133788": "#d62728",
        "FermiNet Q/K/V / 133789": "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    rolled_values = []
    for label, data in series.items():
        steps, values = _windowed(data["step"], data["gap_ev"], start_step=start_step)
        rolled = _plot_raw_and_rolling(
            ax, steps, values, colors[label], rolling_window, label
        )
        rolled_values.append(rolled)
    _set_ylim_from_series(ax, rolled_values, min_padding=0.05)
    ax.set_title(
        f"0097 retry-1 excited-ground gap after step {start_step} "
        f"(trailing {rolling_window}-step mean)"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy-sorted gap (eV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _windowed(
    steps: np.ndarray, values: np.ndarray, *, start_step: int
) -> tuple[np.ndarray, np.ndarray]:
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
    ax.plot(steps, values, color=color, alpha=0.12, linewidth=0.45)
    rolling_steps, rolled = _trailing_rolling_mean(steps, values, rolling_window)
    ax.plot(
        rolling_steps,
        rolled,
        color=color,
        linewidth=2.0,
        label=f"{label} (trailing {rolling_window}-step mean)",
    )
    return rolled


def _trailing_rolling_mean(
    steps: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or len(values) < window:
        return steps, values
    kernel = np.ones(window, dtype=float) / float(window)
    return steps[window - 1 :], np.convolve(values, kernel, mode="valid")


def _set_ylim_from_series(
    ax: plt.Axes, series: list[np.ndarray], *, min_padding: float
) -> None:
    values = np.concatenate([item.reshape(-1) for item in series])
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
