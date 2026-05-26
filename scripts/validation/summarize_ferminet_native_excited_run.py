#!/usr/bin/env python
"""Summarize a native FermiNet excited-state run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    args = _parse_args()
    checkpoint_dir = _resolve_project_path(args.checkpoint_dir)
    output_dir = (
        _resolve_project_path(args.output_dir)
        if args.output_dir
        else checkpoint_dir.parent / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_train_stats(checkpoint_dir / "train_stats.csv")
    energy_mats = _read_appended_npy(checkpoint_dir / "energy_matrix.npy")
    overlap_mats = _read_appended_npy(checkpoint_dir / "overlap_matrix.npy")
    symmetric_overlap_mats = _read_appended_npy(
        checkpoint_dir / "overlap_symmetric_matrix.npy"
    )
    overlap_penalty_mats = _read_appended_npy(
        checkpoint_dir / "overlap_penalty_matrix.npy"
    )
    overlap_scales = _read_appended_npy(
        checkpoint_dir / "overlap_gradient_scale.npy"
    )
    state_orderings = _read_appended_npy(
        checkpoint_dir / "overlap_state_ordering.npy"
    )
    scale_energy_ewm = _read_appended_npy(
        checkpoint_dir / "overlap_scale_energy_ewm.npy"
    )
    scale_std_ewm = _read_appended_npy(checkpoint_dir / "overlap_scale_std_ewm.npy")
    s2_mats = _read_appended_npy(checkpoint_dir / "s2_matrix.npy")
    bare_energy_mats = _read_appended_npy(checkpoint_dir / "bare_energy_matrix.npy")

    summary = _summary(
        rows,
        energy_mats,
        bare_energy_mats,
        overlap_mats,
        symmetric_overlap_mats,
        overlap_penalty_mats,
        overlap_scales,
        state_orderings,
        scale_energy_ewm,
        scale_std_ewm,
        s2_mats,
        experiment=args.experiment,
        job_id=args.job_id,
    )
    json_path = output_dir / "native_ferminet_excited_summary.json"
    md_path = output_dir / "native_ferminet_excited_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(summary), encoding="utf-8")

    print("native_ferminet_excited_summary: ok")
    print(f"checkpoint_dir: {_display_path(checkpoint_dir)}")
    print(f"summary_json: {_display_path(json_path)}")
    print(f"summary_markdown: {_display_path(md_path)}")
    print(f"rows: {summary['rows']}")
    print(f"final_energy: {summary['final_energy']}")
    print(f"final_state_energy: {summary['final_state_energy']}")
    print(f"final_training_state_energy: {summary['final_training_state_energy']}")
    print(f"final_bare_state_energy: {summary['final_bare_state_energy']}")
    print(f"final_bare_gap_hartree: {summary['final_bare_gap_hartree']}")
    print(f"final_overlap_matrix: {summary['final_overlap_matrix']}")
    print(f"final_symmetric_overlap_matrix: {summary['final_symmetric_overlap_matrix']}")
    print(f"final_s2_matrix: {summary['final_s2_matrix']}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="Directory containing train_stats.csv.")
    parser.add_argument("--experiment", default=None, help="Experiment YAML path.")
    parser.add_argument("--job-id", default=None, help="Optional Slurm job id.")
    parser.add_argument("--output-dir", default=None, help="Summary output directory.")
    return parser.parse_args()


def _read_train_stats(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _summary(
    rows: list[dict[str, Any]],
    energy_mats: list[np.ndarray],
    bare_energy_mats: list[np.ndarray],
    overlap_mats: list[np.ndarray],
    symmetric_overlap_mats: list[np.ndarray],
    overlap_penalty_mats: list[np.ndarray],
    overlap_scales: list[np.ndarray],
    state_orderings: list[np.ndarray],
    scale_energy_ewm: list[np.ndarray],
    scale_std_ewm: list[np.ndarray],
    s2_mats: list[np.ndarray],
    *,
    experiment: str | None,
    job_id: str | None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("train_stats.csv has no rows")
    energies = [float(row["energy"]) for row in rows]
    pmoves = [float(row["pmove"]) for row in rows]
    final = rows[-1]
    final_energy_mat = energy_mats[-1] if energy_mats else None
    final_bare_energy_mat = bare_energy_mats[-1] if bare_energy_mats else None
    final_physical_energy_mat = (
        final_bare_energy_mat
        if final_bare_energy_mat is not None
        else final_energy_mat
    )
    final_overlap = overlap_mats[-1] if overlap_mats else None
    final_symmetric_overlap = (
        symmetric_overlap_mats[-1] if symmetric_overlap_mats else None
    )
    final_overlap_penalty = overlap_penalty_mats[-1] if overlap_penalty_mats else None
    final_overlap_scale = overlap_scales[-1] if overlap_scales else None
    final_state_ordering = state_orderings[-1] if state_orderings else None
    final_scale_energy_ewm = scale_energy_ewm[-1] if scale_energy_ewm else None
    final_scale_std_ewm = scale_std_ewm[-1] if scale_std_ewm else None
    final_s2_mat = s2_mats[-1] if s2_mats else None
    if final_bare_energy_mat is not None:
        final_bare_display = final_bare_energy_mat
    elif final_s2_mat is None:
        final_bare_display = final_energy_mat
    else:
        final_bare_display = None
    if final_bare_energy_mat is not None:
        final_bare_gap_source = final_bare_energy_mat
    elif final_s2_mat is None:
        final_bare_gap_source = final_energy_mat
    else:
        final_bare_gap_source = None
    return {
        "status": "ok",
        "experiment": experiment,
        "job_id": job_id,
        "rows": len(rows),
        "first_step": int(rows[0]["step"]),
        "final_step": int(final["step"]),
        "first_energy": energies[0],
        "final_energy": float(final["energy"]),
        "min_energy": min(energies),
        "max_energy": max(energies),
        "final_ewmean": float(final["ewmean"]),
        "final_ewvar": float(final["ewvar"]),
        "mean_pmove": sum(pmoves) / len(pmoves),
        "min_pmove": min(pmoves),
        "max_pmove": max(pmoves),
        "energy_matrix_frames": len(energy_mats),
        "final_state_energy": None
        if final_physical_energy_mat is None
        else _jsonable_array(final_physical_energy_mat),
        "final_training_state_energy": None
        if final_energy_mat is None
        else _jsonable_array(final_energy_mat),
        "final_training_gap_hartree": _state_gap(final_energy_mat),
        "final_training_gap_ev": _hartree_to_ev(_state_gap(final_energy_mat)),
        "bare_energy_matrix_frames": len(bare_energy_mats),
        "final_bare_state_energy": None
        if final_bare_display is None
        else _jsonable_array(final_bare_display),
        "final_bare_gap_hartree": _state_gap(final_bare_gap_source),
        "final_bare_gap_ev": _hartree_to_ev(_state_gap(final_bare_gap_source)),
        "overlap_matrix_frames": len(overlap_mats),
        "final_overlap_matrix": None
        if final_overlap is None
        else _jsonable_array(final_overlap),
        "symmetric_overlap_matrix_frames": len(symmetric_overlap_mats),
        "final_symmetric_overlap_matrix": None
        if final_symmetric_overlap is None
        else _jsonable_array(final_symmetric_overlap),
        "overlap_penalty_matrix_frames": len(overlap_penalty_mats),
        "final_overlap_penalty_matrix": None
        if final_overlap_penalty is None
        else _jsonable_array(final_overlap_penalty),
        "overlap_gradient_scale_frames": len(overlap_scales),
        "final_overlap_gradient_scale": None
        if final_overlap_scale is None
        else _jsonable_array(final_overlap_scale),
        "state_ordering_frames": len(state_orderings),
        "final_state_ordering": None
        if final_state_ordering is None
        else _jsonable_array(final_state_ordering),
        "overlap_scale_energy_ewm_frames": len(scale_energy_ewm),
        "final_overlap_scale_energy_ewm": None
        if final_scale_energy_ewm is None
        else _jsonable_array(final_scale_energy_ewm),
        "overlap_scale_std_ewm_frames": len(scale_std_ewm),
        "final_overlap_scale_std_ewm": None
        if final_scale_std_ewm is None
        else _jsonable_array(final_scale_std_ewm),
        "s2_matrix_frames": len(s2_mats),
        "final_s2_matrix": None
        if final_s2_mat is None
        else _jsonable_array(final_s2_mat),
        "final_s2_diagonal": None
        if final_s2_mat is None
        else _matrix_diagonal(final_s2_mat),
        "final_s2_trace": None
        if final_s2_mat is None
        else _matrix_trace(final_s2_mat),
    }


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Native FermiNet Excited-State Summary",
        "",
        "```text",
        f"status: {summary['status']}",
        f"experiment: {summary['experiment']}",
        f"job_id: {summary['job_id']}",
        f"rows: {summary['rows']}",
        f"steps: {summary['first_step']} -> {summary['final_step']}",
        f"energy: {summary['first_energy']:.12g} -> {summary['final_energy']:.12g}",
        f"energy_range: [{summary['min_energy']:.12g}, {summary['max_energy']:.12g}]",
        f"final_ewmean: {summary['final_ewmean']:.12g}",
        f"final_ewvar: {summary['final_ewvar']:.12g}",
        f"mean_pmove: {summary['mean_pmove']:.12g}",
        f"pmove_range: [{summary['min_pmove']:.12g}, {summary['max_pmove']:.12g}]",
        f"energy_matrix_frames: {summary['energy_matrix_frames']}",
        f"final_state_energy: {summary['final_state_energy']}",
        f"final_training_state_energy: {summary['final_training_state_energy']}",
        f"final_training_gap_hartree: {summary['final_training_gap_hartree']}",
        f"final_training_gap_ev: {summary['final_training_gap_ev']}",
        f"bare_energy_matrix_frames: {summary['bare_energy_matrix_frames']}",
        f"final_bare_state_energy: {summary['final_bare_state_energy']}",
        f"final_bare_gap_hartree: {summary['final_bare_gap_hartree']}",
        f"final_bare_gap_ev: {summary['final_bare_gap_ev']}",
        f"overlap_matrix_frames: {summary['overlap_matrix_frames']}",
        f"final_overlap_matrix: {summary['final_overlap_matrix']}",
        f"symmetric_overlap_matrix_frames: {summary['symmetric_overlap_matrix_frames']}",
        f"final_symmetric_overlap_matrix: {summary['final_symmetric_overlap_matrix']}",
        f"overlap_penalty_matrix_frames: {summary['overlap_penalty_matrix_frames']}",
        f"final_overlap_penalty_matrix: {summary['final_overlap_penalty_matrix']}",
        f"overlap_gradient_scale_frames: {summary['overlap_gradient_scale_frames']}",
        f"final_overlap_gradient_scale: {summary['final_overlap_gradient_scale']}",
        f"state_ordering_frames: {summary['state_ordering_frames']}",
        f"final_state_ordering: {summary['final_state_ordering']}",
        f"overlap_scale_energy_ewm_frames: {summary['overlap_scale_energy_ewm_frames']}",
        f"final_overlap_scale_energy_ewm: {summary['final_overlap_scale_energy_ewm']}",
        f"overlap_scale_std_ewm_frames: {summary['overlap_scale_std_ewm_frames']}",
        f"final_overlap_scale_std_ewm: {summary['final_overlap_scale_std_ewm']}",
        f"s2_matrix_frames: {summary['s2_matrix_frames']}",
        f"final_s2_matrix: {summary['final_s2_matrix']}",
        f"final_s2_diagonal: {summary['final_s2_diagonal']}",
        f"final_s2_trace: {summary['final_s2_trace']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def _jsonable_array(value: Any) -> Any:
    array = np.real_if_close(np.asarray(value))
    if np.iscomplexobj(array):
        return _complex_array_to_json(array)
    return array.tolist()


def _complex_array_to_json(array: np.ndarray) -> Any:
    if array.shape == ():
        scalar = array.item()
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    return [_complex_array_to_json(item) for item in array]


def _matrix_diagonal(matrix: np.ndarray) -> Any:
    array = np.asarray(matrix)
    if array.ndim < 2:
        return None
    return _jsonable_array(np.diag(array))


def _matrix_trace(matrix: np.ndarray) -> Any:
    array = np.asarray(matrix)
    if array.ndim < 2:
        return None
    return _jsonable_array(np.trace(array))


def _state_gap(state_energy: np.ndarray | None) -> float | None:
    if state_energy is None:
        return None
    vector = np.asarray(state_energy)
    if vector.ndim == 2:
        vector = np.diag(vector)
    if vector.ndim != 1 or vector.shape[0] < 2:
        return None
    return float(np.real_if_close(vector[1] - vector[0]))


def _hartree_to_ev(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 27.211386245988


def _resolve_project_path(path: str | Path) -> Path:
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
