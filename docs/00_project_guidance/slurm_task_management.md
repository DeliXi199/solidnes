# SLURM Task Management

Last updated: 2026-05-21

## Purpose

SolidNES uses SLURM for CPU and GPU jobs. The scheduling style follows the
FIIR project:

```text
plan first -> inspect plan -> submit with sbatch only when requested
```

The scheduler should not be a pile of one-off `sbatch` commands. It should
leave a deterministic plan and make CPU/GPU routing explicit.

## File Layout

Scheduling policy:

```text
src/solidnes/slurm_scheduling.py
```

Planner CLI:

```text
scripts/slurm/plan_slurm_job.py
```

Submit wrappers:

```text
scripts/slurm/submit_deepsolid_cpu_smoke.sh
scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

SLURM job templates:

```text
scripts/slurm/run_deepsolid_cpu_smoke.slurm
scripts/slurm/run_deepsolid_gpu_smoke.slurm
```

Plans and logs:

```text
tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/<NNNN_short_slug>/outputs/slurm_plans/plan.json
tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/<NNNN_short_slug>/logs/slurm/
```

## CPU Policy

Default CPU partition order:

```text
regular256 -> regular128 -> regular6430 -> regular -> test
```

Policy:

1. If the job is short (`SOLIDNES_EXPECTED_MINUTES <= 30`) and `test` has an
   idle node, use `test`.
2. Otherwise choose the first idle partition in the policy order.
3. If no policy partition is idle, queue on all policy partitions.
4. If node details are available, request the selected node's full CPU count.
5. CPU jobs are exclusive by default.

Dry-run CPU plan:

```bash
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
```

Submit CPU smoke:

```bash
bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
```

## GPU Policy

Default GPU behavior:

```text
queue_mode: auto
precision_profile: fp64
blocked_partitions: gpu80gllm,gpu40gllm,h20llm,gpu4090,gpu4090_8,gpu4090_128,h800,test
queue_request_when_full: 4 GPUs, 64 CPU cores
```

Policy:

1. Discover GPU nodes from SLURM.
2. Exclude blocked partitions. By default, do not use the LLM mirror
   partitions `gpu80gllm`, `gpu40gllm`, `h20llm`, any RTX 4090 GPU
   partitions (`gpu4090`, `gpu4090_8`, `gpu4090_128`), `h800`, or `test`.
3. Score every currently usable node by `free GPU count * NQS FP64 GPU score`.
   The default single-GPU NQS scores are `H200=100`, `A100 80GB=29`,
   `A100 40GB=29`, `RTX 4090=4`, `H20=3`, `H800=3`, and `generic=1`.
4. Select the highest-scoring usable node. If it is completely idle, pin to it
   and request all GPUs/CPU cores with an exclusive allocation. If it is
   partially occupied, pin to it and request all currently free GPUs/CPU cores
   without exclusive allocation.
5. If all usable nodes are full, queue flexibly on the normal allowed GPU
   partitions with `gpu:4` and `64` CPU cores. The flexible queue excludes
   `intelgpu80g` because that partition has only 2 GPUs per node.
6. The `test` GPU partition is blocked by default. Use it only for tiny
   feasibility checks by explicitly setting `SOLIDNES_GPU_ALLOW_TEST=1` and a
   test-sized resource request.

Dry-run GPU plan:

```bash
SOLIDNES_DRY_RUN=1 bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

Submit GPU smoke:

```bash
bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

## Default Environment

CPU smoke defaults to:

```text
SOLIDNES_CONDA_ENV=solidnes-deepsolid-jax0430-probe
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

GPU smoke defaults to:

```text
SOLIDNES_CONDA_ENV=solidnes-deepsolid-jax0430-cuda12-probe
SOLIDNES_DEEPSOLID_ROOT=external/deepsolid
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Override these when running a production backend or a different DeepSolid
checkout. Do not use `/tmp` for submitted jobs because compute nodes may not
share the login node's temporary directory.

## Execution Status

Actual submission status as of 2026-05-21:

```text
CPU smoke: passed after moving DeepSolid checkout to external/deepsolid
GPU smoke: passed after creating a CUDA-enabled JAX probe environment
GPU one-step Adam smoke: passed after adding a smoke-only checkpoint save shim
GPU adapter-object probe: passed after adding SOLIDNES_BACKEND_SCRIPT override
```

Job IDs:

```text
120610 CPU failed: /tmp DeepSolid checkout not visible on compute node
120615 CPU passed: DeepSolid process smoke completed
120612 GPU failed: JAX did not report a GPU device
120634 GPU passed: JAX reported cuda(id=0), DeepSolid process smoke completed
120654 GPU one-step failed: training stats written, checkpoint save failed
120655 GPU one-step passed: DeepSolid process smoke completed
120686 GPU adapter-object probe passed: initialized objects, local energy, MCMC
```

Keep the CPU and CUDA probe environments separate. Do not overwrite the working
CPU probe environment.

## Which Jobs Go Where

CPU:

- Environment checks.
- Config construction.
- Zero-iteration smoke.
- Small PySCF/DeepSolid debugging.
- Documentation-producing analysis scripts.

GPU:

- One-iteration or longer DeepSolid training smoke.
- Any realistic ground-state VMC run.
- Any future NES-VMC optimization.
- Batch sweeps that include JAX model execution.

## Safety Rules

- Always dry-run a new job shape first.
- Any iterative training or evaluation job with `iterations >= 1000` must write
  a checkpoint for the final training step. For FermiNet-style `qmcjax`
  checkpoints, an `N`-iteration run is complete only when
  `qmcjax_ckpt_{N-1}.npz` exists under that run's checkpoint directory.
- New plans and logs must use the task-bundle layout from
  `docs/00_project_guidance/result_output_numbering.md`.
- Dry-run planning passed on 2026-05-21 for both CPU and GPU smoke wrappers.
- Keep generated plans under the task bundle's `outputs/slurm_plans/`.
- Keep SLURM logs under the task bundle's `logs/slurm/`.
- Do not submit production NES-VMC jobs until the corresponding config is
  reviewed and recorded in `PROGRESS.md`.
- Do not use the smoke-only DeepSolid compatibility shim as evidence that
  production KFAC training works.

## Next Scheduler Step

The current submitter can run one-step smoke by overriding
`SOLIDNES_EXPERIMENT`, and adapter probes by overriding
`SOLIDNES_BACKEND_SCRIPT`; no separate SLURM template is needed yet. Add a new
template only when the resource shape diverges from the existing runtime smoke
wrapper.
