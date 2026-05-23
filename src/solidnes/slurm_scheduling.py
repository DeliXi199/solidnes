"""SLURM scheduling helpers for SolidNES workflows.

The policy mirrors the FIIR project style: build a deterministic plan first,
then submit with `sbatch` only when the caller explicitly asks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shlex
import subprocess
from typing import Any, Iterable, Sequence


CPU_PARTITION_ORDER_DEFAULT = ("regular256", "regular128", "regular6430", "regular", "test")
GPU_BLOCKED_PARTITIONS_DEFAULT = ("gpu80gllm", "gpu40gllm", "h20llm", "gpu4090", "gpu4090_8", "gpu4090_128", "h800", "test")
GPU_FLEXIBLE_QUEUE_PARTITIONS_DEFAULT = ("h200", "amdgpu80g", "amdgpu40g", "h20")
GPU_FLEXIBLE_QUEUE_EXCLUDED_PARTITIONS_DEFAULT = ("intelgpu80g",)
GPU_IDLE_EXTRA_PARTITIONS_DEFAULT = ("intelgpu80g", "amdgpu40g", "amdgpu80g")
GPU_A100_80GB_PARTITIONS = ("intelgpu80g", "amdgpu80g", "gpu80gllm")
GPU_A100_40GB_PARTITIONS = ("amdgpu40g", "gpu40gllm")


@dataclass(frozen=True)
class CpuSchedulingConfig:
    partition_order: tuple[str, ...] = CPU_PARTITION_ORDER_DEFAULT
    expected_minutes: int = 30
    short_task_minutes: int = 30
    test_partition: str = "test"
    force_partition: str = ""
    default_partition_cores: int = 64
    partition_core_overrides: dict[str, int] = field(default_factory=lambda: {"regular": 56})
    exclusive: bool = True
    queue_all_when_no_idle: bool = True


@dataclass(frozen=True)
class GpuSchedulingConfig:
    allowed_partitions: tuple[str, ...] = ()
    blocked_partitions: tuple[str, ...] = GPU_BLOCKED_PARTITIONS_DEFAULT
    idle_extra_partitions: tuple[str, ...] = GPU_IDLE_EXTRA_PARTITIONS_DEFAULT
    allowed_gpu_counts: tuple[int, ...] = ()
    min_gpus: int = 1
    target_gpus: int = 0
    min_cpus: int = 1
    min_memory_mb: int = 0
    queue_min_gpus: int = 4
    queue_min_cpus: int = 64
    queue_memory_mb: int = 64_000
    queue_mode: str = "auto"
    precision_profile: str = "fp64"
    test_partition: str = "test"
    test_partition_max_minutes: int = 30
    reserved_nodes: tuple[str, ...] = ()
    exclusive_when_full_node: bool = False

    def __post_init__(self) -> None:
        queue_mode = self.queue_mode.strip().lower()
        if queue_mode not in {"auto", "pinned", "flexible"}:
            raise ValueError("GPU queue_mode must be auto, pinned, or flexible")
        object.__setattr__(self, "queue_mode", queue_mode)
        object.__setattr__(self, "allowed_partitions", normalize_partitions(self.allowed_partitions))
        object.__setattr__(self, "blocked_partitions", normalize_partitions(self.blocked_partitions))
        object.__setattr__(self, "idle_extra_partitions", normalize_partitions(self.idle_extra_partitions))
        object.__setattr__(self, "reserved_nodes", normalize_partitions(self.reserved_nodes))
        object.__setattr__(self, "allowed_gpu_counts", normalize_gpu_counts(self.allowed_gpu_counts))
        if self.min_gpus < 1 or self.queue_min_gpus < 1:
            raise ValueError("GPU counts must be positive")
        if self.target_gpus < 0:
            raise ValueError("target_gpus must be non-negative")
        if self.min_cpus < 1 or self.queue_min_cpus < 1:
            raise ValueError("CPU counts must be positive")
        if self.allowed_gpu_counts and not any(count >= self.min_gpus for count in self.allowed_gpu_counts):
            raise ValueError("allowed_gpu_counts must include at least one value >= min_gpus")


@dataclass(frozen=True)
class CpuNode:
    name: str
    partitions: tuple[str, ...]
    state: str
    total_cpus: int
    allocated_cpus: int = 0
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    gres: str = "(null)"

    @property
    def idle_cpus(self) -> int:
        return max(0, self.total_cpus - self.allocated_cpus)

    @property
    def is_idle(self) -> bool:
        return _state_allows_work(self.state) and "IDLE" in self.state.upper() and self.idle_cpus > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "partitions": list(self.partitions),
            "state": self.state,
            "total_cpus": self.total_cpus,
            "allocated_cpus": self.allocated_cpus,
            "idle_cpus": self.idle_cpus,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "gres": self.gres,
            "is_idle": self.is_idle,
        }


@dataclass(frozen=True)
class GpuNode:
    name: str
    partitions: tuple[str, ...]
    state: str
    total_cpus: int
    allocated_cpus: int
    total_memory_mb: int
    free_memory_mb: int
    gres: str
    gpu_type: str
    total_gpus: int
    allocated_gpus: int = 0

    @property
    def idle_cpus(self) -> int:
        return max(0, self.total_cpus - self.allocated_cpus)

    @property
    def free_gpus(self) -> int:
        return max(0, self.total_gpus - self.allocated_gpus)

    @property
    def is_idle(self) -> bool:
        return (
            _state_allows_work(self.state)
            and "IDLE" in self.state.upper()
            and self.idle_cpus >= self.total_cpus
            and self.free_gpus >= self.total_gpus
        )

    @property
    def is_available(self) -> bool:
        return _state_allows_work(self.state) and self.free_gpus > 0

    @property
    def gpu_model_key(self) -> str:
        text = " ".join((self.gpu_type, *self.partitions)).lower()
        partitions = set(self.partitions)
        if partitions.intersection(GPU_A100_80GB_PARTITIONS) or ("a100" in text and "80" in text):
            return "a100_80gb"
        if partitions.intersection(GPU_A100_40GB_PARTITIONS) or ("a100" in text and "40" in text):
            return "a100_40gb"
        for key in ("h200", "h800", "h20", "4090", "amd", "intel"):
            if key in text:
                return "rtx4090" if key == "4090" else key
        return self.gpu_type.lower() or "generic"

    def gres_request(self, count: int | None = None) -> str:
        gpu_count = self.free_gpus if count is None else count
        if self.gpu_type:
            return f"gpu:{self.gpu_type}:{gpu_count}"
        return f"gpu:{gpu_count}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "partitions": list(self.partitions),
            "state": self.state,
            "total_cpus": self.total_cpus,
            "allocated_cpus": self.allocated_cpus,
            "idle_cpus": self.idle_cpus,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "gres": self.gres,
            "gpu_type": self.gpu_type,
            "gpu_model_key": self.gpu_model_key,
            "total_gpus": self.total_gpus,
            "allocated_gpus": self.allocated_gpus,
            "free_gpus": self.free_gpus,
            "is_idle": self.is_idle,
            "is_available": self.is_available,
        }


def discover_cpu_nodes() -> list[CpuNode]:
    scontrol = _run_command(["scontrol", "show", "node"])
    if scontrol.returncode == 0:
        nodes = parse_scontrol_cpu_nodes(scontrol.stdout)
        if nodes:
            return nodes
    sinfo = _run_command(["sinfo", "-N", "-h", "-o", "%N|%P|%T|%C|%m|%G"])
    if sinfo.returncode == 0:
        return parse_sinfo_cpu_nodes(sinfo.stdout)
    return []


def discover_gpu_nodes() -> list[GpuNode]:
    scontrol = _run_command(["scontrol", "show", "node"])
    if scontrol.returncode == 0:
        nodes = parse_scontrol_gpu_nodes(scontrol.stdout)
        if nodes:
            return nodes
    sinfo = _run_command(["sinfo", "-N", "-h", "-o", "%N|%P|%T|%C|%m|%G"])
    if sinfo.returncode == 0:
        return parse_sinfo_gpu_nodes(sinfo.stdout)
    return []


def parse_scontrol_cpu_nodes(text: str) -> list[CpuNode]:
    nodes = []
    for fields in _scontrol_field_blocks(text):
        total_cpus = _positive_int(fields.get("CPUTot")) or _positive_int(parse_tres(fields.get("CfgTRES", "")).get("cpu"))
        if total_cpus <= 0:
            continue
        nodes.append(
            CpuNode(
                name=fields.get("NodeName", ""),
                partitions=parse_partition_list([fields.get("Partitions", "")]),
                state=fields.get("State", ""),
                total_cpus=total_cpus,
                allocated_cpus=_positive_int(fields.get("CPUAlloc")),
                total_memory_mb=_memory_to_mb(fields.get("RealMemory", "")),
                free_memory_mb=_memory_to_mb(fields.get("FreeMem", "")),
                gres=fields.get("Gres", "(null)"),
            )
        )
    return sorted([node for node in nodes if node.name], key=lambda node: node.name)


def parse_scontrol_gpu_nodes(text: str) -> list[GpuNode]:
    nodes = []
    for fields in _scontrol_field_blocks(text):
        gres = fields.get("Gres", "")
        gpu_type, gres_gpus = parse_gres(gres)
        cfg_tres = parse_tres(fields.get("CfgTRES", ""))
        alloc_tres = parse_tres(fields.get("AllocTRES", ""))
        total_gpus = _positive_int(cfg_tres.get("gres/gpu")) or gres_gpus
        if total_gpus <= 0:
            continue
        nodes.append(
            GpuNode(
                name=fields.get("NodeName", ""),
                partitions=parse_partition_list([fields.get("Partitions", "")]),
                state=fields.get("State", ""),
                total_cpus=_positive_int(fields.get("CPUTot")) or _positive_int(cfg_tres.get("cpu")),
                allocated_cpus=_positive_int(fields.get("CPUAlloc")) or _positive_int(alloc_tres.get("cpu")),
                total_memory_mb=_memory_to_mb(fields.get("RealMemory", "")) or _memory_to_mb(cfg_tres.get("mem", "")),
                free_memory_mb=_memory_to_mb(fields.get("FreeMem", "")),
                gres=gres,
                gpu_type=gpu_type,
                total_gpus=total_gpus,
                allocated_gpus=_positive_int(alloc_tres.get("gres/gpu")),
            )
        )
    return sorted([node for node in nodes if node.name], key=lambda node: node.name)


def parse_sinfo_cpu_nodes(text: str) -> list[CpuNode]:
    by_node: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        name, partition, state, cpus, memory, gres = (line.split("|") + [""] * 6)[:6]
        allocated, _idle, _other, total = parse_sinfo_cpu_counts(cpus)
        if total <= 0:
            continue
        entry = by_node.setdefault(
            name,
            {"partitions": [], "state": state, "total": total, "allocated": allocated, "memory": _memory_to_mb(memory), "gres": gres},
        )
        normalized = normalize_partition_name(partition)
        if normalized and normalized not in entry["partitions"]:
            entry["partitions"].append(normalized)
    return [
        CpuNode(name, tuple(item["partitions"]), item["state"], item["total"], item["allocated"], item["memory"], item["memory"], item["gres"])
        for name, item in sorted(by_node.items())
    ]


def parse_sinfo_gpu_nodes(text: str) -> list[GpuNode]:
    nodes = []
    for cpu_node in parse_sinfo_cpu_nodes(text):
        gpu_type, total_gpus = parse_gres(cpu_node.gres)
        if total_gpus <= 0:
            continue
        allocated = 0 if cpu_node.is_idle else total_gpus
        nodes.append(
            GpuNode(
                name=cpu_node.name,
                partitions=cpu_node.partitions,
                state=cpu_node.state,
                total_cpus=cpu_node.total_cpus,
                allocated_cpus=cpu_node.allocated_cpus,
                total_memory_mb=cpu_node.total_memory_mb,
                free_memory_mb=cpu_node.free_memory_mb,
                gres=cpu_node.gres,
                gpu_type=gpu_type,
                total_gpus=total_gpus,
                allocated_gpus=allocated,
            )
        )
    return nodes


def plan_slurm_job(
    *,
    kind: str,
    cpu_nodes: Sequence[CpuNode] | None = None,
    gpu_nodes: Sequence[GpuNode] | None = None,
    cpu_config: CpuSchedulingConfig | None = None,
    gpu_config: GpuSchedulingConfig | None = None,
    job_name: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    log_dir: str = "logs/slurm",
    submit_script: str | None = None,
) -> dict[str, Any]:
    if kind == "cpu":
        return plan_cpu_job(cpu_nodes if cpu_nodes is not None else discover_cpu_nodes(), cpu_config or CpuSchedulingConfig(), job_name, account, time_limit, log_dir, submit_script)
    if kind == "gpu":
        return plan_gpu_job(gpu_nodes if gpu_nodes is not None else discover_gpu_nodes(), gpu_config or GpuSchedulingConfig(), job_name, account, time_limit, log_dir, submit_script)
    raise ValueError(f"unknown SLURM job kind: {kind}")


def plan_cpu_job(
    nodes: Sequence[CpuNode],
    config: CpuSchedulingConfig,
    job_name: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    log_dir: str = "logs/slurm",
    submit_script: str | None = None,
) -> dict[str, Any]:
    partition_order = normalize_partitions(config.partition_order)
    force_partition = normalize_partition_name(config.force_partition)
    candidates = _cpu_candidates(nodes, partition_order)
    selected_partition = ""
    selected_node = None
    reason = "no_idle_cpu_partition"

    if force_partition:
        selected_partition = force_partition
        selected_node = _best_cpu_node(nodes, force_partition)
        reason = "forced"
    elif config.expected_minutes <= config.short_task_minutes and _best_cpu_node(nodes, config.test_partition):
        selected_partition = normalize_partition_name(config.test_partition)
        selected_node = _best_cpu_node(nodes, selected_partition)
        reason = "short_job_idle_test"
    else:
        for partition in partition_order:
            selected_node = _best_cpu_node(nodes, partition)
            if selected_node:
                selected_partition = partition
                reason = "first_idle_in_policy_order"
                break

    core_budget = None
    if selected_partition:
        core_budget = selected_node.total_cpus if selected_node else config.partition_core_overrides.get(selected_partition, config.default_partition_cores)
        partition_arg = selected_partition
    elif config.queue_all_when_no_idle:
        partition_arg = ",".join(partition_order)
        reason = "no_idle_queue_all_policy_partitions"
    else:
        partition_arg = ""

    if not partition_arg:
        return {"kind": "cpu", "ready": False, "reason": reason, "selection": {"candidates": candidates}, "sbatch": None}

    args = ["--nodes", "1"]
    if config.exclusive:
        args.append("--exclusive")
    args.extend(["--partition", partition_arg])
    if job_name:
        args.extend(["--job-name", job_name])
    if account:
        args.extend(["--account", account])
    if core_budget is not None:
        args.extend(["--ntasks-per-node", str(core_budget), "--export", f"ALL,SOLIDNES_TOTAL_CPU_CORES={core_budget}"])
    else:
        args.extend(["--export", "ALL"])
    if time_limit:
        args.extend(["--time", time_limit])
    args.extend(["--output", f"{log_dir}/%x_%j.log", "--error", f"{log_dir}/%x_%j.err"])
    command = ["sbatch", *args] + ([submit_script] if submit_script else [])
    return {
        "kind": "cpu",
        "ready": True,
        "reason": reason,
        "config": {"partition_order": list(partition_order), "expected_minutes": config.expected_minutes},
        "selection": {
            "partition_arg": partition_arg,
            "selected_partition": selected_partition,
            "selected_node": selected_node.to_dict() if selected_node else None,
            "core_budget": core_budget,
            "candidates": candidates,
        },
        "sbatch": _sbatch_dict(args, command, {"partition": partition_arg, "cpus": core_budget, "gpus": 0}),
    }


def plan_gpu_job(
    nodes: Sequence[GpuNode],
    config: GpuSchedulingConfig,
    job_name: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    log_dir: str = "logs/slurm",
    submit_script: str | None = None,
) -> dict[str, Any]:
    eligible = _eligible_gpu_nodes(nodes, config)
    chosen = sorted(
        eligible,
        key=lambda node: (
            -_gpu_score(node, config),
            -_requestable_gpu_count_for_node(node, config),
            -node.idle_cpus,
            -node.free_memory_mb,
            node.name,
        ),
    )
    selected = chosen[0] if chosen else None
    partition_arg = ""
    reason = "no_eligible_gpu_node"
    nodelist: str | None = None
    gres = f"gpu:{config.queue_min_gpus}"
    cpus = config.queue_min_cpus
    memory = config.queue_memory_mb
    requested_gpus = config.queue_min_gpus
    exclusive = False

    if config.queue_mode in {"auto", "pinned"} and selected is not None:
        selected_partition = _selected_gpu_partition(selected, config) or selected.partitions[0]
        partition_arg = selected_partition
        reason = "best_idle_gpu_node" if selected.is_idle else "best_partial_gpu_node"
        nodelist = selected.name
        requested_gpus = _requested_gpus_for_selected_node(selected, config)
        gres = selected.gres_request(requested_gpus)
        cpus = _requested_cpus_for_selected_node(selected, config, requested_gpus)
        memory = config.min_memory_mb
        exclusive = selected.is_idle and requested_gpus >= selected.total_gpus
    elif config.queue_mode in {"auto", "flexible"}:
        partition_arg = ",".join(_flexible_gpu_partitions(config))
        reason = "no_free_gpu_queue_flexible_partitions"
    if not partition_arg:
        return {"kind": "gpu", "ready": False, "reason": reason, "selection": {"candidates": [node.to_dict() for node in eligible]}, "sbatch": None}

    args = ["--nodes", "1", "--partition", partition_arg, "--gres", gres, "--cpus-per-task", str(cpus)]
    if exclusive:
        args.append("--exclusive")
    if nodelist:
        args.extend(["--nodelist", nodelist])
    if memory > 0:
        args.extend(["--mem", str(memory)])
    if job_name:
        args.extend(["--job-name", job_name])
    if account:
        args.extend(["--account", account])
    if time_limit:
        args.extend(["--time", time_limit])
    args.extend(
        [
            "--export",
            _export_arg(_gpu_cpu_exports(cpus, requested_gpus)),
            "--output",
            f"{log_dir}/%x_%j.log",
            "--error",
            f"{log_dir}/%x_%j.err",
        ]
    )
    command = ["sbatch", *args] + ([submit_script] if submit_script else [])
    return {
        "kind": "gpu",
        "ready": True,
        "reason": reason,
        "config": {
            "allowed_partitions": list(config.allowed_partitions),
            "blocked_partitions": list(config.blocked_partitions),
            "idle_extra_partitions": list(config.idle_extra_partitions),
            "allowed_gpu_counts": list(config.allowed_gpu_counts),
            "min_gpus": config.min_gpus,
            "target_gpus": config.target_gpus,
            "min_cpus": config.min_cpus,
            "queue_min_gpus": config.queue_min_gpus,
            "queue_min_cpus": config.queue_min_cpus,
            "queue_mode": config.queue_mode,
            "precision_profile": config.precision_profile,
        },
        "selection": {
            "selected_partition": partition_arg,
            "selected_node": selected.to_dict() if selected else None,
            "candidates": [node.to_dict() for node in chosen],
        },
        "sbatch": _sbatch_dict(args, command, {"partition": partition_arg, "cpus": cpus, "gpus": requested_gpus, "gres": gres}),
    }


def _gpu_cpu_exports(total_cpus: int, total_gpus: int) -> dict[str, str]:
    total_cpus = max(1, int(total_cpus))
    total_gpus = max(1, int(total_gpus))
    base = total_cpus // total_gpus
    remainder = total_cpus % total_gpus
    sizes = [base + (1 if gpu_index < remainder else 0) for gpu_index in range(total_gpus)]

    batches = []
    start = 0
    for gpu_index, size in enumerate(sizes):
        if size <= 0:
            batches.append(f"gpu{gpu_index}:none")
            continue
        end = start + size - 1
        cpu_range = str(start) if start == end else f"{start}-{end}"
        batches.append(f"gpu{gpu_index}:{cpu_range}")
        start = end + 1

    return {
        "SOLIDNES_TOTAL_GPUS": str(total_gpus),
        "SOLIDNES_TOTAL_CPU_CORES": str(total_cpus),
        "SOLIDNES_CPUS_PER_GPU_BASE": str(base),
        "SOLIDNES_CPU_REMAINDER": str(remainder),
        "SOLIDNES_CPUS_PER_GPU_LIST": ":".join(str(size) for size in sizes),
        "SOLIDNES_CPU_BATCHES": ";".join(batches),
        "SOLIDNES_CPU_PARALLELISM": str(total_cpus),
    }


def _requested_gpus_for_selected_node(node: GpuNode, config: GpuSchedulingConfig) -> int:
    """Choose the GPU count after a concrete node has been selected."""
    return _requestable_gpu_count_for_node(node, config)


def _requested_cpus_for_selected_node(node: GpuNode, config: GpuSchedulingConfig, requested_gpus: int) -> int:
    if node.is_idle and requested_gpus >= node.total_gpus:
        return node.idle_cpus
    cpus_per_gpu = max(1, config.min_cpus // max(1, config.min_gpus))
    requested_cpus = max(config.min_cpus, requested_gpus * cpus_per_gpu)
    return min(node.idle_cpus, requested_cpus)


def _export_arg(values: dict[str, str]) -> str:
    assignments = [f"{key}={value}" for key, value in values.items()]
    return ",".join(["ALL", *assignments])


def parse_gres(gres: str) -> tuple[str, int]:
    match = re.search(r"gpu(?::([^:,\s]+))?:(\d+)", gres or "")
    if not match:
        return "", 0
    return match.group(1) or "", int(match.group(2))


def parse_tres(text: str) -> dict[str, str]:
    result = {}
    for item in str(text or "").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def parse_sinfo_cpu_counts(text: str) -> tuple[int, int, int, int]:
    values = [int(match.group(0)) for match in re.finditer(r"\d+", str(text))]
    if len(values) >= 4:
        return values[0], values[1], values[2], values[3]
    return 0, 0, 0, values[0] if values else 0


def parse_partition_list(values: Iterable[str]) -> tuple[str, ...]:
    return normalize_partitions(values)


def normalize_partition_name(partition: str) -> str:
    return str(partition).strip().rstrip("*")


def normalize_partitions(partitions: Iterable[str]) -> tuple[str, ...]:
    result = []
    for raw in partitions:
        for item in str(raw).replace(",", " ").split():
            normalized = normalize_partition_name(item)
            if normalized and normalized not in result:
                result.append(normalized)
    return tuple(result)


def normalize_gpu_counts(counts: Iterable[int | str]) -> tuple[int, ...]:
    result = []
    for raw in counts:
        for item in str(raw).replace(",", " ").split():
            count = int(item)
            if count > 0 and count not in result:
                result.append(count)
    return tuple(sorted(result))


def shell_assignments(values: dict[str, str]) -> str:
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in sorted(values.items()))


def _eligible_gpu_nodes(nodes: Sequence[GpuNode], config: GpuSchedulingConfig) -> list[GpuNode]:
    result = []
    for node in nodes:
        if node.name in config.reserved_nodes:
            continue
        selectable_partitions = [
            partition
            for partition in node.partitions
            if partition not in config.blocked_partitions
            and (not config.allowed_partitions or partition in config.allowed_partitions or partition in config.idle_extra_partitions)
        ]
        if not selectable_partitions:
            continue
        if not node.is_available:
            continue
        if node.free_gpus < config.min_gpus or node.idle_cpus < config.min_cpus:
            continue
        if _requestable_gpu_count_for_node(node, config) < config.min_gpus:
            continue
        if config.min_memory_mb and node.total_memory_mb and node.total_memory_mb < config.min_memory_mb:
            continue
        result.append(node)
    return result


def _gpu_score(node: GpuNode, config: GpuSchedulingConfig) -> int:
    fp64 = {
        "h200": 100,
        "a100_80gb": 29,
        "a100_40gb": 29,
        "rtx4090": 4,
        "h20": 3,
        "h800": 3,
        "generic": 1,
    }
    tf32 = {
        "h200": 989,
        "h800": 989,
        "a100_80gb": 156,
        "a100_40gb": 156,
        "rtx4090": 100,
        "h20": 74,
        "amd": 70,
        "intel": 40,
        "generic": 80,
    }
    weights = fp64 if config.precision_profile == "fp64" else tf32
    return weights.get(node.gpu_model_key, weights["generic"]) * max(1, _requestable_gpu_count_for_node(node, config))


def _requestable_gpu_count_for_node(node: GpuNode, config: GpuSchedulingConfig) -> int:
    target_gpus = config.target_gpus if config.target_gpus > 0 else node.free_gpus
    requested = min(node.free_gpus, target_gpus)
    if config.allowed_gpu_counts:
        allowed = [
            count
            for count in config.allowed_gpu_counts
            if config.min_gpus <= count <= requested
        ]
        return max(allowed) if allowed else 0
    return max(config.min_gpus, requested)


def _selected_gpu_partition(node: GpuNode, config: GpuSchedulingConfig) -> str:
    for partition in node.partitions:
        if partition not in config.blocked_partitions and (
            not config.allowed_partitions or partition in config.allowed_partitions or partition in config.idle_extra_partitions
        ):
            return partition
    return ""


def _flexible_gpu_partitions(config: GpuSchedulingConfig) -> tuple[str, ...]:
    base = config.allowed_partitions or GPU_FLEXIBLE_QUEUE_PARTITIONS_DEFAULT
    return tuple(
        partition
        for partition in base
        if partition not in config.blocked_partitions and partition not in GPU_FLEXIBLE_QUEUE_EXCLUDED_PARTITIONS_DEFAULT
    )


def _cpu_candidates(nodes: Sequence[CpuNode], partition_order: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "partition": partition,
            "best_idle_node": (_best_cpu_node(nodes, partition).to_dict() if _best_cpu_node(nodes, partition) else None),
        }
        for partition in partition_order
    ]


def _best_cpu_node(nodes: Sequence[CpuNode], partition: str) -> CpuNode | None:
    eligible = [node for node in nodes if normalize_partition_name(partition) in node.partitions and node.is_idle]
    if not eligible:
        return None
    return sorted(eligible, key=lambda node: (-node.idle_cpus, -node.free_memory_mb, node.name))[0]


def _scontrol_field_blocks(text: str) -> list[dict[str, str]]:
    blocks = []
    current = []
    for line in text.splitlines():
        if line.startswith("NodeName=") and current:
            blocks.append("\n".join(current))
            current = [line]
        elif line.strip() or current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [
        {match.group(1): match.group(2) for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", block)}
        for block in blocks
    ]


def _memory_to_mb(value: str | None) -> int:
    if not value:
        return 0
    match = re.match(r"^(\d+)([KMGTP]?)", str(value).strip(), re.IGNORECASE)
    if not match:
        return 0
    number = int(match.group(1))
    unit = match.group(2).upper()
    if unit == "K":
        return max(1, number // 1024)
    if unit == "G":
        return number * 1024
    if unit == "T":
        return number * 1024 * 1024
    return number


def _positive_int(value: str | int | None) -> int:
    if value is None:
        return 0
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def _state_allows_work(state: str) -> bool:
    blocked = ("DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESP", "POWER", "RESV")
    upper = state.upper()
    return not any(marker in upper for marker in blocked)


def _sbatch_dict(args: list[str], command: list[str], request: dict[str, Any]) -> dict[str, Any]:
    return {
        "sbatch_args": args,
        "sbatch_command": command,
        "sbatch_command_quoted": " ".join(shlex.quote(part) for part in command),
        "request": request,
    }


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr=str(exc))
