# SolidNES SLURM Scripts

Submit through the policy wrappers from the repository root.

CPU smoke:

```bash
SOLIDNES_TASK_ROOT=tasks/.../<task_type>/<NNNN_short_slug> \
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
SOLIDNES_TASK_ROOT=tasks/.../<task_type>/<NNNN_short_slug> \
bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
```

GPU smoke:

```bash
SOLIDNES_TASK_ROOT=tasks/.../<task_type>/<NNNN_short_slug> \
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
SOLIDNES_TASK_ROOT=tasks/.../<task_type>/<NNNN_short_slug> \
bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

The wrappers call:

```text
scripts/slurm/plan_slurm_job.py
```

and write deterministic plans under:

```text
tasks/.../<task_type>/<NNNN_short_slug>/outputs/slurm_plans/plan.json
```

Useful overrides:

- `SOLIDNES_TASK_ROOT`: numbered task bundle root. Submit wrappers require this
  unless both `SOLIDNES_PLAN_JSON` and `SOLIDNES_SLURM_LOG_DIR` are set.
- `SOLIDNES_DRY_RUN=1`: print and save the plan without `sbatch`.
- `SOLIDNES_PLAN_JSON`: task-bundle plan path. New runs should use
  `tasks/.../<task_type>/<NNNN_short_slug>/outputs/slurm_plans/plan.json`.
- `SOLIDNES_SLURM_LOG_DIR`: task-bundle log directory. New runs should use
  `tasks/.../<task_type>/<NNNN_short_slug>/logs/slurm/`.
- `SOLIDNES_JOB_NAME`: include the run number, for example
  `0001_ferminet_pbc_hf_pretrain_gpu_pilot`.
- `SOLIDNES_CONDA_ENV`: conda environment. CPU smoke defaults to
  `solidnes-deepsolid-jax0430-probe`; GPU smoke defaults to
  `solidnes-deepsolid-jax0430-cuda12-probe`.
- `SOLIDNES_BACKEND_SCRIPT`: backend runner script, default
  `scripts/backends/run_deepsolid_process_smoke.py`.
- `SOLIDNES_EXPERIMENT`: experiment YAML passed to the DeepSolid smoke runner.
- `SOLIDNES_DEEPSOLID_ROOT`: external DeepSolid checkout, default `external/deepsolid`.
- `SOLIDNES_TIME_LIMIT`: SLURM wall time, default `00:30:00`.
- `SOLIDNES_SLURM_ACCOUNT`: SLURM account, default `hmt03`.
- `SOLIDNES_FORCE_PARTITION`: force one CPU partition.
- `SOLIDNES_GPU_PARTITIONS`: GPU allowlist, default `auto`.
- `SOLIDNES_GPU_BLOCKED_PARTITIONS`: GPU partitions to skip. Defaults to the
  LLM mirror partitions, all RTX 4090 GPU partitions, `h800`, and `test`:
  `gpu80gllm,gpu40gllm,h20llm,gpu4090,gpu4090_8,gpu4090_128,h800,test`.
- `SOLIDNES_GPU_ALLOW_TEST=1`: remove `test` from the default GPU blocked list
  for tiny feasibility checks only.
- `SOLIDNES_GPU_PRECISION_PROFILE`: `fp64` or `tf32`, default `fp64`.

Direct `sbatch scripts/slurm/*.slurm` also works, but wrappers are preferred
because they record the scheduling plan before submission.

In `auto` GPU mode, the planner scores currently usable nodes by free GPU count
times the SolidNES NQS FP64 GPU score. It pins to the strongest usable node and
requests all currently free GPUs/CPU cores there. Completely idle nodes use an
exclusive allocation; partially occupied nodes do not. When no usable GPU is
available now, the flexible queue requests `gpu:4` plus 64 CPU cores across the
normal allowed partitions, excluding `intelgpu80g` because it has only two GPUs
per node and excluding `h800` because it is not available to the default
SolidNES account.

## Current Status

CPU smoke has passed through SLURM with:

```text
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
```

GPU smoke has passed through SLURM with:

```text
SOLIDNES_CONDA_ENV=solidnes-deepsolid-jax0430-cuda12-probe
job 120634
COMPLETED 0:0
jax_default_backend=gpu
jax_devices=[cuda(id=0)]
```

GPU adapter-object probe also passed through the same wrapper by setting
`SOLIDNES_BACKEND_SCRIPT=scripts/backends/probe_deepsolid_adapter_objects.py`:

```text
job 120686
COMPLETED 0:0
node gpu40904
```
