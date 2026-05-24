#!/usr/bin/env python
"""Summarize FermiNet training progress from stderr step logs.

This is a fallback for timed-out SLURM jobs where train_stats.csv exists but
was not flushed before cancellation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


INITIAL_RE = re.compile(r"Initial energy: (?P<energy>[-+0-9.eE]+) E_h")
STEP_RE = re.compile(
    r"Step (?P<step>\d+): (?P<energy>[-+0-9.eE]+) E_h, "
    r"exp\. variance=(?P<variance>[-+0-9.eE]+) E_h\^2, "
    r"pmove=(?P<pmove>[-+0-9.eE]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--err", required=True, help="SLURM stderr log.")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--tail-rows", type=int, default=100)
    args = parser.parse_args()

    err_path = Path(args.err).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    text = err_path.read_text(encoding="utf-8", errors="replace")
    rows = _parse_steps(text)
    if not rows:
        raise ValueError(f"No FermiNet step rows found in {err_path}")

    initial_match = INITIAL_RE.search(text)
    initial_energy = (
        float(initial_match.group("energy")) if initial_match is not None else None
    )
    timeout = "DUE TO TIME LIMIT" in text
    tail_rows = rows[-min(args.tail_rows, len(rows)) :]
    block_means = _block_means([row["energy"] for row in tail_rows], 5)
    summary: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": args.job_id,
        "source_log": str(err_path),
        "timed_out": timeout,
        "initial_energy_hartree": initial_energy,
        "rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "first_energy_hartree": rows[0]["energy"],
        "last_energy_hartree": rows[-1]["energy"],
        "min_energy_hartree": min(row["energy"] for row in rows),
        "min_energy_step": min(rows, key=lambda row: row["energy"])["step"],
        "energy_delta_hartree": rows[-1]["energy"] - rows[0]["energy"],
        "last_exp_variance_hartree2": rows[-1]["variance"],
        "pmove_mean": _mean(row["pmove"] for row in rows),
        "pmove_last": rows[-1]["pmove"],
        "pmove_min": min(row["pmove"] for row in rows),
        "pmove_max": max(row["pmove"] for row in rows),
        "tail_rows": len(tail_rows),
        "tail_energy_mean_hartree": _mean(row["energy"] for row in tail_rows),
        "tail_energy_stderr_hartree": _std_error(row["energy"] for row in tail_rows),
        "tail_exp_variance_mean_hartree2": _mean(
            row["variance"] for row in tail_rows
        ),
        "tail_pmove_mean": _mean(row["pmove"] for row in tail_rows),
        "block_energy_means_hartree": block_means,
        "block_energy_stderr_hartree": _std_error(block_means),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, summary)
    print(f"rows: {summary['rows']}")
    print(f"last_step: {summary['last_step']}")
    print(f"last_energy_hartree: {summary['last_energy_hartree']}")
    print(f"tail_energy_mean_hartree: {summary['tail_energy_mean_hartree']}")
    print(f"tail_pmove_mean: {summary['tail_pmove_mean']}")
    print(f"timed_out: {summary['timed_out']}")
    print(f"summary_json: {output_json}")
    print(f"summary_md: {output_md}")
    return 0


def _parse_steps(text: str) -> list[dict[str, float]]:
    rows = []
    for match in STEP_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group("step")),
                "energy": float(match.group("energy")),
                "variance": float(match.group("variance")),
                "pmove": float(match.group("pmove")),
            }
        )
    return rows


def _mean(values: Any) -> float:
    return float(statistics.fmean(list(values)))


def _std_error(values: Any) -> float | None:
    values_list = list(values)
    if len(values_list) < 2:
        return None
    return float(statistics.stdev(values_list) / math.sqrt(len(values_list)))


def _block_means(values: list[float], block_count: int) -> list[float]:
    if len(values) < block_count:
        return []
    block_size = len(values) // block_count
    return [
        _mean(values[start : start + block_size])
        for start in range(0, block_size * block_count, block_size)
    ]


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# FermiNet Training Log Summary",
        "",
        f"- Job id: `{summary['job_id']}`",
        f"- Source log: `{summary['source_log']}`",
        f"- Timed out: `{summary['timed_out']}`",
        f"- Parsed rows: `{summary['rows']}`",
        f"- Step range: `{summary['first_step']}` to `{summary['last_step']}`",
        f"- Initial energy: `{summary['initial_energy_hartree']}` Ha",
        f"- First logged energy: `{summary['first_energy_hartree']}` Ha",
        f"- Last logged energy: `{summary['last_energy_hartree']}` Ha",
        f"- Minimum energy: `{summary['min_energy_hartree']}` Ha at step `{summary['min_energy_step']}`",
        f"- Energy delta: `{summary['energy_delta_hartree']}` Ha",
        f"- Last exp variance: `{summary['last_exp_variance_hartree2']}` Ha^2",
        f"- Mean pmove: `{summary['pmove_mean']}`",
        f"- Last pmove: `{summary['pmove_last']}`",
        f"- Tail rows: `{summary['tail_rows']}`",
        f"- Tail energy mean: `{summary['tail_energy_mean_hartree']}` Ha",
        f"- Tail energy stderr: `{summary['tail_energy_stderr_hartree']}` Ha",
        f"- Tail exp variance mean: `{summary['tail_exp_variance_mean_hartree2']}` Ha^2",
        f"- Tail pmove mean: `{summary['tail_pmove_mean']}`",
        f"- Block energy means: `{summary['block_energy_means_hartree']}` Ha",
        f"- Block energy stderr: `{summary['block_energy_stderr_hartree']}` Ha",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
