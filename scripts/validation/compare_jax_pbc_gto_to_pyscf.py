#!/usr/bin/env python
"""Compare the minimal JAX PBC GTO evaluator against PySCF PBC."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (SRC_ROOT, FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from solidnes.backends.ferminet_configs.diamond_pbc_gamma_paper import (  # noqa: E402
    diamond_primitive_atoms,
    diamond_primitive_lattice,
)
from solidnes.backends.ferminet_pbc_gto import PbcGtoEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--image-cutoff", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--precision", type=float, default=1e-10)
    parser.add_argument("--exp-to-discard", type=float, default=0.1)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--top-columns", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    cell = build_diamond_cell(
        basis=args.basis,
        precision=args.precision,
        exp_to_discard=args.exp_to_discard,
    )
    kpts = np.zeros((1, 3), dtype=np.float64)
    coords = sample_cell_points(cell.a, args.num_points, args.seed)

    ao_pyscf = first_k(cell.eval_gto("PBCGTOval_sph", coords, kpts=kpts))
    evaluator = PbcGtoEvaluator.from_pyscf_cell(
        cell,
        image_cutoff=args.image_cutoff,
        kpt=kpts[0],
    )
    ao_jax = np.asarray(evaluator.eval_ao(jnp.asarray(coords)))
    ao_label_info = make_ao_label_info(cell)
    ao_metrics = diff_metrics(
        ao_jax,
        ao_pyscf,
        column_labels=ao_label_info["labels"],
        group_keys=ao_label_info["shell_keys"],
        top_columns=args.top_columns,
    )

    mo_metrics = None
    hf_summary = None
    if not args.skip_hf:
        mean_field = pbc_scf.KRHF(cell, exxdiv="ewald", kpts=kpts).density_fit()
        mean_field.conv_tol = 1e-10
        hf_energy = float(mean_field.kernel(mean_field.get_init_guess()))
        mo_coeff = first_k(mean_field.mo_coeff)
        mo_occ = first_k_vector(mean_field.mo_occ)
        occupied = np.asarray(mo_occ) > 0.9
        mo_pyscf = ao_pyscf @ mo_coeff[:, occupied]
        mo_jax = ao_jax @ mo_coeff[:, occupied]
        mo_metrics = diff_metrics(
            mo_jax,
            mo_pyscf,
            column_labels=[
                f"occupied_mo_{idx}" for idx in range(int(np.count_nonzero(occupied)))
            ],
            top_columns=args.top_columns,
        )
        hf_summary = {
            "energy_hartree": hf_energy,
            "converged": bool(mean_field.converged),
            "occupied_orbitals": int(np.count_nonzero(occupied)),
        }

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "system": "diamond_c_primitive_gamma",
        "basis": args.basis,
        "image_cutoff": args.image_cutoff,
        "num_points": args.num_points,
        "seed": args.seed,
        "cell": {
            "nao": int(cell.nao_nr()),
            "nelectron": int(cell.nelectron),
            "volume_bohr3": float(cell.vol),
            "lattice_rows_bohr": np.asarray(cell.a).tolist(),
            "ao_labels": ao_label_info["labels"],
        },
        "ao": ao_metrics,
        "mo": mo_metrics,
        "hf": hf_summary,
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else PROJECT_ROOT
        / "outputs"
        / "validation"
        / f"jax_pbc_gto_{args.basis}_cutoff{args.image_cutoff}_summary.json"
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else output_json.with_suffix(".md")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, summary)
    print_summary(summary, output_json, output_md)
    return 0


def build_diamond_cell(
    *,
    basis: str,
    precision: float,
    exp_to_discard: float,
) -> Any:
    lattice = diamond_primitive_lattice()
    atoms = diamond_primitive_atoms(lattice)
    cell = pbc_gto.Cell()
    cell.atom = [[atom.symbol, atom.coords] for atom in atoms]
    cell.a = np.asarray(lattice, dtype=np.float64).T
    cell.unit = "B"
    cell.basis = basis
    cell.spin = 0
    cell.verbose = 0
    cell.precision = precision
    cell.exp_to_discard = exp_to_discard
    cell.build()
    return cell


def sample_cell_points(lattice: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fractional = rng.random((num_points, 3))
    return fractional @ np.asarray(lattice, dtype=np.float64)


def first_k(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value[0])
    array = np.asarray(value)
    if array.ndim >= 3:
        return array[0]
    return array


def first_k_vector(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value[0])
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    return array


def make_ao_label_info(cell: Any) -> dict[str, list[str]]:
    labels = [label.strip() for label in cell.ao_labels()]
    shell_keys = []
    for atom, symbol, orbital, component in cell.ao_labels(fmt=False):
        angular = "".join(ch for ch in orbital if ch.isalpha())
        shell_keys.append(f"atom{atom}:{symbol}:{orbital}:{angular}")
        if component:
            labels[len(shell_keys) - 1] = f"{labels[len(shell_keys) - 1]} ({component})"
    return {"labels": labels, "shell_keys": shell_keys}


def diff_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    column_labels: list[str] | None = None,
    group_keys: list[str] | None = None,
    top_columns: int = 8,
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    abs_diff = np.abs(diff)
    expected_abs = np.abs(expected)
    denom = np.maximum(expected_abs, 1e-12)
    column_summary = per_column_metrics(
        abs_diff,
        expected_abs,
        column_labels=column_labels,
        group_keys=group_keys,
        top_columns=top_columns,
    )
    return {
        "shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs": float(np.max(abs_diff)),
        "rms_abs": float(np.sqrt(np.mean(abs_diff**2))),
        "max_rel": float(np.max(abs_diff / denom)),
        "mean_abs_expected": float(np.mean(expected_abs)),
        "max_abs_expected": float(np.max(expected_abs)),
        "finite_checks_passed": bool(
            np.all(np.isfinite(np.real(actual)))
            and np.all(np.isfinite(np.imag(actual)))
            and np.all(np.isfinite(np.real(expected)))
            and np.all(np.isfinite(np.imag(expected)))
        ),
        **column_summary,
    }


def per_column_metrics(
    abs_diff: np.ndarray,
    expected_abs: np.ndarray,
    *,
    column_labels: list[str] | None,
    group_keys: list[str] | None,
    top_columns: int,
) -> dict[str, Any]:
    if abs_diff.ndim < 2:
        return {}
    ncols = abs_diff.shape[-1]
    flat_abs = abs_diff.reshape(-1, ncols)
    flat_expected = expected_abs.reshape(-1, ncols)
    if column_labels is None:
        column_labels = [f"column_{idx}" for idx in range(ncols)]
    column_max = np.max(flat_abs, axis=0)
    column_rms = np.sqrt(np.mean(flat_abs**2, axis=0))
    column_expected_max = np.max(flat_expected, axis=0)
    worst_index = int(np.argmax(column_max))
    order = np.argsort(column_max)[::-1][: max(0, int(top_columns))]
    top = [
        {
            "column": int(idx),
            "label": column_labels[idx] if idx < len(column_labels) else f"column_{idx}",
            "max_abs": float(column_max[idx]),
            "rms_abs": float(column_rms[idx]),
            "max_abs_expected": float(column_expected_max[idx]),
            "relative_to_column_max": float(
                column_max[idx] / max(column_expected_max[idx], 1e-12)
            ),
        }
        for idx in order
    ]
    summary: dict[str, Any] = {
        "worst_column": {
            "column": worst_index,
            "label": (
                column_labels[worst_index]
                if worst_index < len(column_labels)
                else f"column_{worst_index}"
            ),
            "max_abs": float(column_max[worst_index]),
            "rms_abs": float(column_rms[worst_index]),
            "max_abs_expected": float(column_expected_max[worst_index]),
            "relative_to_column_max": float(
                column_max[worst_index] / max(column_expected_max[worst_index], 1e-12)
            ),
        },
        "top_columns_by_max_abs": top,
    }
    if group_keys:
        summary["groups_by_max_abs"] = group_metrics(
            column_max,
            column_rms,
            column_expected_max,
            group_keys=group_keys,
        )
    return summary


def group_metrics(
    column_max: np.ndarray,
    column_rms: np.ndarray,
    column_expected_max: np.ndarray,
    *,
    group_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for idx, key in enumerate(group_keys):
        grouped.setdefault(key, []).append(idx)
    rows = []
    for key, indices in grouped.items():
        idxs = np.asarray(indices, dtype=np.int64)
        max_abs = float(np.max(column_max[idxs]))
        expected_max = float(np.max(column_expected_max[idxs]))
        rows.append(
            {
                "group": key,
                "columns": [int(idx) for idx in indices],
                "max_abs": max_abs,
                "max_rms_abs": float(np.max(column_rms[idxs])),
                "max_abs_expected": expected_max,
                "relative_to_group_max": max_abs / max(expected_max, 1e-12),
            }
        )
    return sorted(rows, key=lambda row: row["max_abs"], reverse=True)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    ao = summary["ao"]
    mo = summary.get("mo")
    hf = summary.get("hf") or {}
    lines = [
        "# JAX PBC GTO vs PySCF",
        "",
        f"Created: `{summary['created_at']}`",
        f"System: `{summary['system']}`",
        f"Basis: `{summary['basis']}`",
        f"Image cutoff: `{summary['image_cutoff']}`",
        f"Sample points: `{summary['num_points']}`",
        "",
        "## Cell",
        "",
        f"- NAO: {summary['cell']['nao']}",
        f"- Electrons: {summary['cell']['nelectron']}",
        f"- Volume: {summary['cell']['volume_bohr3']:.6f} bohr^3",
        "",
        "## AO",
        "",
        f"- Shape: `{ao['shape']}`",
        f"- Max abs: {ao['max_abs']:.12g}",
        f"- RMS abs: {ao['rms_abs']:.12g}",
        f"- Max rel: {ao['max_rel']:.12g}",
        f"- Worst column: `{ao['worst_column']['label']}` "
        f"(index {ao['worst_column']['column']}, "
        f"max abs {ao['worst_column']['max_abs']:.12g})",
        f"- Finite checks: `{ao['finite_checks_passed']}`",
        "",
        "### AO Top Columns",
        "",
        "| Column | Label | Max abs | RMS abs | Rel to col max |",
        "| ---: | --- | ---: | ---: | ---: |",
        *[
            f"| {row['column']} | `{row['label']}` | {row['max_abs']:.12g} | "
            f"{row['rms_abs']:.12g} | {row['relative_to_column_max']:.12g} |"
            for row in ao.get("top_columns_by_max_abs", [])
        ],
        "",
        "### AO Top Shell Groups",
        "",
        "| Group | Max abs | Max RMS abs | Rel to group max |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| `{row['group']}` | {row['max_abs']:.12g} | "
            f"{row['max_rms_abs']:.12g} | {row['relative_to_group_max']:.12g} |"
            for row in ao.get("groups_by_max_abs", [])[:8]
        ],
        "",
    ]
    if mo:
        lines += [
            "## Occupied MO",
            "",
            f"- Shape: `{mo['shape']}`",
            f"- Max abs: {mo['max_abs']:.12g}",
            f"- RMS abs: {mo['rms_abs']:.12g}",
            f"- Max rel: {mo['max_rel']:.12g}",
            f"- Worst column: `{mo['worst_column']['label']}` "
            f"(index {mo['worst_column']['column']}, "
            f"max abs {mo['worst_column']['max_abs']:.12g})",
            f"- Finite checks: `{mo['finite_checks_passed']}`",
            "",
            "## HF",
            "",
            f"- Converged: `{hf.get('converged')}`",
            f"- Occupied orbitals: {hf.get('occupied_orbitals')}",
            f"- Energy: {hf.get('energy_hartree'):.12g} Ha",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    ao = summary["ao"]
    mo = summary.get("mo") or {}
    print(f"basis: {summary['basis']}")
    print(f"image_cutoff: {summary['image_cutoff']}")
    print(f"ao_max_abs: {ao['max_abs']:.12g}")
    print(f"ao_rms_abs: {ao['rms_abs']:.12g}")
    if mo:
        print(f"mo_max_abs: {mo['max_abs']:.12g}")
        print(f"mo_rms_abs: {mo['rms_abs']:.12g}")
    print(f"summary_json: {output_json}")
    print(f"summary_md: {output_md}")


if __name__ == "__main__":
    raise SystemExit(main())
