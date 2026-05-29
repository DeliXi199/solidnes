#!/usr/bin/env python
"""Plan CPU or GPU SLURM jobs for SolidNES."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solidnes.slurm_scheduling import (  # noqa: E402
    CPU_PARTITION_ORDER_DEFAULT,
    GPU_BLOCKED_PARTITIONS_DEFAULT,
    CpuSchedulingConfig,
    GpuSchedulingConfig,
    normalize_partitions,
    parse_scontrol_cpu_nodes,
    parse_scontrol_gpu_nodes,
    parse_sinfo_cpu_nodes,
    parse_sinfo_gpu_nodes,
    plan_slurm_job,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a SolidNES SLURM job.")
    parser.add_argument("--kind", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--job-name", default="")
    parser.add_argument("--account", default="")
    parser.add_argument("--time", default="")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--submit-script", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--run-sbatch", action="store_true")

    cpu = parser.add_argument_group("CPU policy")
    cpu.add_argument("--partition-order", default=" ".join(CPU_PARTITION_ORDER_DEFAULT))
    cpu.add_argument("--expected-minutes", type=int, default=30)
    cpu.add_argument("--short-task-minutes", type=int, default=30)
    cpu.add_argument("--test-partition", default="test")
    cpu.add_argument("--force-partition", default="")

    gpu = parser.add_argument_group("GPU policy")
    gpu.add_argument("--partition", action="append", default=[])
    gpu.add_argument("--blocked-partition", action="append", default=None)
    gpu.add_argument("--allowed-gpu-count", action="append", default=[])
    gpu.add_argument("--min-gpus", type=int, default=1)
    gpu.add_argument("--target-gpus", type=int, default=0)
    gpu.add_argument("--min-cpus", type=int, default=1)
    gpu.add_argument("--min-memory-mb", type=int, default=0)
    gpu.add_argument("--queue-min-gpus", type=int, default=4)
    gpu.add_argument("--queue-min-cpus", type=int, default=64)
    gpu.add_argument("--queue-memory-mb", type=int, default=64000)
    gpu.add_argument("--gpu-queue-mode", choices=("auto", "pinned", "flexible"), default="auto")
    gpu.add_argument("--precision-profile", choices=("fp64", "tf32"), default="fp64")
    gpu.add_argument("--reserved-node", action="append", default=[])
    gpu.add_argument("--exclusive-when-full-node", action="store_true")

    fixtures = parser.add_argument_group("fixtures")
    fixtures.add_argument("--scontrol-output", default="")
    fixtures.add_argument("--sinfo-output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.run_sbatch and not args.submit_script:
        raise SystemExit("--run-sbatch requires --submit-script")
    plan = _build_plan(args)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    if args.print_json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        _print_summary(plan)
    if args.run_sbatch:
        if not plan.get("ready"):
            raise SystemExit("SLURM plan is not ready")
        completed = subprocess.run(plan["sbatch"]["sbatch_command"], text=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    elif not plan.get("ready"):
        raise SystemExit(1)
    return plan


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    submit_script = args.submit_script or None
    job_name = args.job_name or None
    account = args.account or None
    time_limit = args.time or None
    if args.kind == "cpu":
        nodes = _load_cpu_nodes(args)
        config = CpuSchedulingConfig(
            partition_order=normalize_partitions(args.partition_order.split()),
            expected_minutes=args.expected_minutes,
            short_task_minutes=args.short_task_minutes,
            test_partition=args.test_partition,
            force_partition=args.force_partition,
        )
        return plan_slurm_job(
            kind="cpu",
            cpu_nodes=nodes,
            cpu_config=config,
            job_name=job_name,
            account=account,
            time_limit=time_limit,
            log_dir=args.log_dir,
            submit_script=submit_script,
        )
    nodes = _load_gpu_nodes(args)
    config = GpuSchedulingConfig(
        allowed_partitions=normalize_partitions(args.partition),
        blocked_partitions=normalize_partitions(args.blocked_partition or GPU_BLOCKED_PARTITIONS_DEFAULT),
        allowed_gpu_counts=tuple(args.allowed_gpu_count),
        min_gpus=args.min_gpus,
        target_gpus=args.target_gpus,
        min_cpus=args.min_cpus,
        min_memory_mb=args.min_memory_mb,
        queue_min_gpus=args.queue_min_gpus,
        queue_min_cpus=args.queue_min_cpus,
        queue_memory_mb=args.queue_memory_mb,
        queue_mode=args.gpu_queue_mode,
        precision_profile=args.precision_profile,
        reserved_nodes=normalize_partitions(args.reserved_node),
        exclusive_when_full_node=args.exclusive_when_full_node,
    )
    return plan_slurm_job(
        kind="gpu",
        gpu_nodes=nodes,
        gpu_config=config,
        job_name=job_name or "solidnes-gpu",
        account=account,
        time_limit=time_limit,
        log_dir=args.log_dir,
        submit_script=submit_script,
    )


def _load_cpu_nodes(args: argparse.Namespace):
    if args.scontrol_output:
        return parse_scontrol_cpu_nodes(Path(args.scontrol_output).read_text(encoding="utf-8"))
    if args.sinfo_output:
        return parse_sinfo_cpu_nodes(Path(args.sinfo_output).read_text(encoding="utf-8"))
    return None


def _load_gpu_nodes(args: argparse.Namespace):
    if args.scontrol_output:
        return parse_scontrol_gpu_nodes(Path(args.scontrol_output).read_text(encoding="utf-8"))
    if args.sinfo_output:
        return parse_sinfo_gpu_nodes(Path(args.sinfo_output).read_text(encoding="utf-8"))
    return None


def _print_summary(plan: dict[str, Any]) -> None:
    print("SolidNES SLURM scheduling policy:")
    print(f"  kind: {plan.get('kind')}")
    print(f"  ready: {plan.get('ready')}")
    print(f"  reason: {plan.get('reason')}")
    if not plan.get("ready"):
        return
    selection = plan["selection"]
    request = plan["sbatch"]["request"]
    print(f"  selected_partition: {request.get('partition')}")
    if plan.get("kind") == "cpu":
        print(f"  selected_core_budget: {request.get('cpus') or 'slurm_default'}")
    else:
        node = selection.get("selected_node") or {}
        print(f"  reference_node: {node.get('name') or 'slurm_assigned_at_runtime'}")
        print(f"  gpu_model: {node.get('gpu_model_key') or 'slurm_assigned_at_runtime'}")
        print(f"  gres: {request.get('gres')}")
    print(f"  sbatch: {plan['sbatch']['sbatch_command_quoted']}")


if __name__ == "__main__":
    main()
