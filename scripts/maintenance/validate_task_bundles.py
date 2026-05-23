#!/usr/bin/env python
"""Validate SolidNES numbered task-bundle structure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = PROJECT_ROOT / "tasks"
RUN_INDEX = PROJECT_ROOT / "records" / "run_index.md"
TASK_LEDGER = TASKS_ROOT / "TASK_LEDGER.md"
MIGRATION_MANIFEST = TASKS_ROOT / "MIGRATION_MANIFEST.tsv"
TASK_RE = re.compile(r"^(?P<run>\d{4})_.+")


def _iter_task_dirs(tasks_root: Path) -> list[Path]:
    task_dirs: list[Path] = []
    for path in tasks_root.rglob("*"):
        if not path.is_dir():
            continue
        if TASK_RE.match(path.name):
            task_dirs.append(path)
    return sorted(task_dirs)


def _read_manifest_runs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {row["run"] for row in csv.DictReader(handle, delimiter="\t")}


def _read_markdown_runs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r"^\| (\d{4}) \|", path.read_text(), flags=re.MULTILINE))


def validate(tasks_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    task_dirs = _iter_task_dirs(tasks_root)
    seen: dict[str, Path] = {}
    for task_dir in task_dirs:
        match = TASK_RE.match(task_dir.name)
        if match is None:
            continue
        run = match.group("run")
        if run in seen:
            errors.append(f"duplicate run {run}: {seen[run]} and {task_dir}")
        seen[run] = task_dir
        for child in ("logs", "outputs", "results"):
            if not (task_dir / child).is_dir():
                errors.append(f"missing {child}/ in {task_dir}")

    manifest_runs = _read_manifest_runs(MIGRATION_MANIFEST)
    ledger_runs = _read_markdown_runs(TASK_LEDGER)
    index_runs = _read_markdown_runs(RUN_INDEX)
    actual_runs = set(seen)

    for label, runs in (
        ("MIGRATION_MANIFEST.tsv", manifest_runs),
        ("TASK_LEDGER.md", ledger_runs),
        ("records/run_index.md", index_runs),
    ):
        stale = sorted(runs - actual_runs)
        if stale:
            warnings.append(f"{label} references missing task runs: {', '.join(stale)}")

    missing_from_ledger = sorted(actual_runs - ledger_runs)
    if missing_from_ledger:
        warnings.append(
            "TASK_LEDGER.md is missing task runs: " + ", ".join(missing_from_ledger)
        )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-root",
        default=str(TASKS_ROOT),
        help="Task root to validate. Defaults to the project tasks/ directory.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).resolve()
    errors, warnings = validate(tasks_root)
    task_count = len(_iter_task_dirs(tasks_root))

    print(f"tasks_root: {tasks_root}")
    print(f"task_count: {task_count}")
    print(f"errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
