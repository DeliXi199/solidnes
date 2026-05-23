# Progress: SLURM Scheduling Scaffold

Date: 2026-05-21

## What Was Added

Added FIIR-style scheduling structure:

```text
src/solidnes/slurm_scheduling.py
scripts/slurm/plan_slurm_job.py
scripts/slurm/run_deepsolid_cpu_smoke.slurm
scripts/slurm/run_deepsolid_gpu_smoke.slurm
scripts/slurm/submit_deepsolid_cpu_smoke.sh
scripts/slurm/submit_deepsolid_gpu_smoke.sh
scripts/slurm/README.md
docs/00_project_guidance/slurm_task_management.md
```

## Policy

CPU:

```text
test for short idle jobs, otherwise regular256 -> regular128 -> regular6430 -> regular -> test
```

GPU:

```text
auto queue mode, fp64 precision profile, block gpu4090/gpu4090_8/h20llm by default
```

## Interpretation

SolidNES now has a task-submission layer for CPU and GPU jobs. The wrappers
create deterministic plan JSON files and only call `sbatch` when not in dry-run
mode.

## Validation

Passed:

```text
python -m py_compile src/solidnes/slurm_scheduling.py scripts/slurm/plan_slurm_job.py
bash -n scripts/slurm/*.sh scripts/slurm/*.slurm
YAML validation
```

Dry-run planning also passed:

```bash
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

Observed dry-run plans:

```text
CPU: queue on regular256,regular128,regular6430,regular,test
GPU: queue on h200,h800,h20,gpu4090_128,test with --gres gpu:1
```

No job was submitted during validation.

## Next Step

Next runnable step:

```bash
bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
```

Submit GPU only after confirming the `solidnes-deepsolid-jax0430-probe`
environment has a CUDA-enabled JAX build on the target node.
