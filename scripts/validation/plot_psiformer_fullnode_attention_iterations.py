#!/usr/bin/env python
"""Plot full-node PsiFormer attention training trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "tasks/psiformer/0096_psiformer_attention_full_stack"
DEFAULT_VARIANTS = {
    "upstream/FermiNet": "fullnode_anygpu_ferminet_b4096_i10000",
    "fused-QKV": "fullnode_anygpu_fused_qkv_b4096_i10000",
}
METRICS = (
    ("energy", "Energy (Ha)"),
    ("ewmean", "EW mean (Ha)"),
    ("ewvar", "EW variance"),
    ("pmove", "Pmove"),
)


def main() -> int:
    args = _parse_args()
    task_root = _resolve_path(args.task_root)
    output_dir = task_root / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    series = {
        label: _read_train_stats(
            task_root / "runs" / run_name / "results/checkpoints/train_stats.csv"
        )
        for label, run_name in DEFAULT_VARIANTS.items()
    }
    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_iteration_comparison_full",
        "PsiFormer attention comparison: full 10000 steps",
        start_step=None,
        rolling_window=100,
    )
    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_iteration_comparison_after1000",
        "PsiFormer attention comparison: steps 1000--9999",
        start_step=1000,
        rolling_window=100,
    )
    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_iteration_comparison_last1000",
        "PsiFormer attention comparison: last 1000 steps",
        start_step=9000,
        rolling_window=25,
    )
    print("psiformer_fullnode_attention_iteration_plots: ok")
    for suffix in ("full", "after1000", "last1000"):
        print(
            "plot: "
            + _display_path(
                output_dir / f"psiformer_fullnode_10k_iteration_comparison_{suffix}.png"
            )
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    return parser.parse_args()


def _read_train_stats(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data: dict[str, np.ndarray] = {"step": np.array([int(row["step"]) for row in rows])}
    for key, _label in METRICS:
        data[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return data


def _plot_window(
    series: dict[str, dict[str, np.ndarray]],
    output_base: Path,
    title: str,
    *,
    start_step: int | None,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {
        "upstream/FermiNet": "#1f77b4",
        "fused-QKV": "#d62728",
    }
    for ax, (metric, ylabel) in zip(axes.ravel(), METRICS, strict=True):
        for label, data in series.items():
            mask = np.ones_like(data["step"], dtype=bool)
            if start_step is not None:
                mask = data["step"] >= start_step
            steps = data["step"][mask]
            values = data[metric][mask]
            color = colors.get(label)
            ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
            if len(values) >= rolling_window:
                rolled = _rolling_mean(values, rolling_window)
                ax.plot(
                    steps,
                    rolled,
                    color=color,
                    linewidth=1.7,
                    label=f"{label} ({rolling_window}-step mean)",
                )
            else:
                ax.plot(steps, values, color=color, linewidth=1.7, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 1].set_xlabel("Iteration")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "svg"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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
