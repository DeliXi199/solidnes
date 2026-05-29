#!/usr/bin/env python
"""Plot PsiFormer full-node ground/excited state energy gaps."""

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
DEFAULT_TASK_ROOT = PROJECT_ROOT / "tasks/psiformer/0096_psiformer_attention_full_stack"
DEFAULT_VARIANTS = {
    "upstream/FermiNet": "fullnode_anygpu_ferminet_b4096_i10000",
    "fused-QKV": "fullnode_anygpu_fused_qkv_b4096_i10000",
}
HARTREE_TO_EV = 27.211386245988


def main() -> int:
    args = _parse_args()
    task_root = _resolve_path(args.task_root)
    output_dir = task_root / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    series = {
        label: _read_variant(task_root / "runs" / run_name / "results/checkpoints")
        for label, run_name in DEFAULT_VARIANTS.items()
    }
    summary = _summarize(series, rolling_window=args.rolling_window)

    timeseries_csv = output_dir / "psiformer_fullnode_10k_state_gap_timeseries.csv"
    summary_csv = output_dir / "psiformer_fullnode_10k_state_gap_summary.csv"
    summary_json = output_dir / "psiformer_fullnode_10k_state_gap_summary.json"
    _write_timeseries_csv(timeseries_csv, series)
    _write_summary_csv(summary_csv, summary)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_state_energy_gap_comparison_full",
        "PsiFormer state energies and gap: full 10000 steps",
        start_step=None,
        rolling_window=args.rolling_window,
    )
    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_state_energy_gap_comparison_after1000",
        "PsiFormer state energies and gap: steps 1000--9999",
        start_step=1000,
        rolling_window=args.rolling_window,
    )
    _plot_window(
        series,
        output_dir / "psiformer_fullnode_10k_state_energy_gap_comparison_last1000",
        "PsiFormer state energies and gap: last 1000 steps",
        start_step=9000,
        rolling_window=25,
    )

    print("psiformer_fullnode_state_gap_plots: ok")
    print(f"timeseries_csv: {_display_path(timeseries_csv)}")
    print(f"summary_csv: {_display_path(summary_csv)}")
    print(f"summary_json: {_display_path(summary_json)}")
    for suffix in ("full", "after1000", "last1000"):
        print(
            "plot: "
            + _display_path(
                output_dir
                / f"psiformer_fullnode_10k_state_energy_gap_comparison_{suffix}.png"
            )
        )
    for item in summary:
        print(
            f"{item['variant']}: final_gap={item['final_gap_hartree']:.9f} Ha "
            f"({item['final_gap_ev']:.6f} eV), "
            f"tail1000_gap_mean={item['tail1000_gap_hartree_mean']:.9f} Ha "
            f"({item['tail1000_gap_ev_mean']:.6f} eV)"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--rolling-window", type=int, default=100)
    return parser.parse_args()


def _read_variant(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    steps = _read_steps(checkpoint_dir / "train_stats.csv")
    raw_energy = _stack_vectors(_read_appended_npy(checkpoint_dir / "energy_matrix.npy"))
    if raw_energy.ndim != 2 or raw_energy.shape[1] < 2:
        raise ValueError(f"expected at least two states in {checkpoint_dir}")
    if len(steps) != raw_energy.shape[0]:
        steps = np.arange(raw_energy.shape[0], dtype=int)

    orderings = _stack_vectors(
        _read_appended_npy(checkpoint_dir / "overlap_state_ordering.npy"),
        dtype=int,
    )
    if orderings.shape != raw_energy.shape:
        orderings = np.argsort(raw_energy, axis=1)

    energy_order = np.argsort(raw_energy, axis=1)
    row_index = np.arange(raw_energy.shape[0])
    ordered_by_energy = raw_energy[row_index[:, None], energy_order]
    ordered_by_state_ordering = raw_energy[row_index[:, None], orderings]

    return {
        "step": steps,
        "raw_energy": raw_energy,
        "state_ordering": orderings,
        "ordered_by_energy": ordered_by_energy,
        "ordered_by_state_ordering": ordered_by_state_ordering,
        "ground_energy": ordered_by_energy[:, 0],
        "excited_energy": ordered_by_energy[:, -1],
        "gap_hartree": ordered_by_energy[:, -1] - ordered_by_energy[:, 0],
        "gap_ev": (ordered_by_energy[:, -1] - ordered_by_energy[:, 0])
        * HARTREE_TO_EV,
    }


def _read_steps(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        return np.array([int(row["step"]) for row in csv.DictReader(handle)], dtype=int)


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


def _stack_vectors(
    arrays: list[np.ndarray],
    *,
    dtype: type[float] | type[int] = float,
) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0), dtype=dtype)
    return np.stack([np.asarray(array, dtype=dtype).reshape(-1) for array in arrays])


def _summarize(
    series: dict[str, dict[str, np.ndarray]],
    *,
    rolling_window: int,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for label, data in series.items():
        gap_hartree = data["gap_hartree"]
        gap_ev = data["gap_ev"]
        state_ordering = data["state_ordering"]
        default_order = np.tile(
            np.arange(state_ordering.shape[1]), (state_ordering.shape[0], 1)
        )
        ordering_swap_steps = data["step"][
            np.any(state_ordering != default_order, axis=1)
        ]
        row: dict[str, Any] = {
            "variant": label,
            "frames": int(len(gap_hartree)),
            "first_step": int(data["step"][0]),
            "final_step": int(data["step"][-1]),
            "first_raw_state0": float(data["raw_energy"][0, 0]),
            "first_raw_state1": float(data["raw_energy"][0, 1]),
            "final_raw_state0": float(data["raw_energy"][-1, 0]),
            "final_raw_state1": float(data["raw_energy"][-1, 1]),
            "first_ground_energy": float(data["ground_energy"][0]),
            "first_excited_energy": float(data["excited_energy"][0]),
            "final_ground_energy": float(data["ground_energy"][-1]),
            "final_excited_energy": float(data["excited_energy"][-1]),
            "first_gap_hartree": float(gap_hartree[0]),
            "first_gap_ev": float(gap_ev[0]),
            "final_gap_hartree": float(gap_hartree[-1]),
            "final_gap_ev": float(gap_ev[-1]),
            "min_gap_hartree": float(np.min(gap_hartree)),
            "min_gap_ev": float(np.min(gap_ev)),
            "max_gap_hartree": float(np.max(gap_hartree)),
            "max_gap_ev": float(np.max(gap_ev)),
            "tail100_gap_hartree_mean": _tail_mean(gap_hartree, 100),
            "tail100_gap_ev_mean": _tail_mean(gap_ev, 100),
            "tail1000_gap_hartree_mean": _tail_mean(gap_hartree, 1000),
            "tail1000_gap_ev_mean": _tail_mean(gap_ev, 1000),
            "rolling_window": rolling_window,
            "final_rolling_gap_hartree": float(
                _rolling_mean(gap_hartree, rolling_window)[-1]
            ),
            "final_rolling_gap_ev": float(_rolling_mean(gap_ev, rolling_window)[-1]),
            "state_ordering_first": data["state_ordering"][0].astype(int).tolist(),
            "state_ordering_final": data["state_ordering"][-1].astype(int).tolist(),
            "state_ordering_swap_count": int(len(ordering_swap_steps)),
            "state_ordering_first_swap_step": None
            if len(ordering_swap_steps) == 0
            else int(ordering_swap_steps[0]),
            "state_ordering_last_swap_step": None
            if len(ordering_swap_steps) == 0
            else int(ordering_swap_steps[-1]),
        }
        summary.append(row)
    return summary


def _write_timeseries_csv(path: Path, series: dict[str, dict[str, np.ndarray]]) -> None:
    fieldnames = [
        "variant",
        "step",
        "raw_state0",
        "raw_state1",
        "state_ordering0",
        "state_ordering1",
        "state_ordered0",
        "state_ordered1",
        "ground_energy",
        "excited_energy",
        "gap_hartree",
        "gap_ev",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, data in series.items():
            for idx, step in enumerate(data["step"]):
                writer.writerow(
                    {
                        "variant": label,
                        "step": int(step),
                        "raw_state0": float(data["raw_energy"][idx, 0]),
                        "raw_state1": float(data["raw_energy"][idx, 1]),
                        "state_ordering0": int(data["state_ordering"][idx, 0]),
                        "state_ordering1": int(data["state_ordering"][idx, 1]),
                        "state_ordered0": float(
                            data["ordered_by_state_ordering"][idx, 0]
                        ),
                        "state_ordered1": float(
                            data["ordered_by_state_ordering"][idx, 1]
                        ),
                        "ground_energy": float(data["ground_energy"][idx]),
                        "excited_energy": float(data["excited_energy"][idx]),
                        "gap_hartree": float(data["gap_hartree"][idx]),
                        "gap_ev": float(data["gap_ev"][idx]),
                    }
                )


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _plot_window(
    series: dict[str, dict[str, np.ndarray]],
    output_base: Path,
    title: str,
    *,
    start_step: int | None,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    colors = {
        "upstream/FermiNet": "#1f77b4",
        "fused-QKV": "#d62728",
    }
    metrics = (
        ("ground_energy", "Ground energy (Ha)"),
        ("excited_energy", "Excited energy (Ha)"),
        ("gap_ev", "Gap (eV)"),
    )
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for label, data in series.items():
            mask = np.ones_like(data["step"], dtype=bool)
            if start_step is not None:
                mask = data["step"] >= start_step
            steps = data["step"][mask]
            values = data[metric][mask]
            color = colors.get(label)
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
            else:
                ax.plot(steps, values, color=color, linewidth=1.8, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "svg"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def _tail_mean(values: np.ndarray, count: int) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values[-min(count, len(values)) :]))


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
