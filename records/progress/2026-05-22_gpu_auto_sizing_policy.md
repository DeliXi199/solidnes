# GPU Auto-Sizing Scheduling Policy

Date: 2026-05-22

## Motivation

The cc-pVDZ continuation submission showed that fixed GPU requests can miss
usable idle nodes. In particular, `intelgpu80g/gpu001` had 2 idle GPUs, but a
4-GPU request could not use it.

## Change

Updated the GPU scheduling path so it checks current available GPU nodes first
and then derives the GPU request from the selected node:

- `min_gpus` is now the hard minimum acceptable GPU count.
- `target_gpus` is the preferred GPU count when enough GPUs are available.
- For a concrete selected node, requested GPUs are:
  `min(node.free_gpus, target_gpus)` with `min_gpus` as the floor.
- If `target_gpus` is `0`, the scheduler uses all free GPUs on the selected
  node.
- If no eligible node is available, the scheduler falls back to the existing
  flexible queue behavior.

For the shell submitter:

- Legacy `SOLIDNES_GPU_MIN_GPUS` is treated as a target/preferred GPU count.
- New `SOLIDNES_GPU_HARD_MIN_GPUS` controls the hard lower bound, default `1`.
- Legacy `SOLIDNES_GPU_MIN_CPUS` is treated as a queued-resource target.
- New `SOLIDNES_GPU_HARD_MIN_CPUS` controls the hard CPU lower bound, default
  `8`.

## Verification

Constructed-node validation:

```text
input node: intelgpu80g/gpu001, IDLE, free_gpus=2, idle_cpus=96
config: min_gpus=1, target_gpus=8
selected reason: best_idle_gpu_node
requested resources: partition=intelgpu80g, gpus=2, cpus=96, gres=gpu:2
```

Submitter dry-run with legacy-style `SOLIDNES_GPU_MIN_GPUS=8` now prints:

```text
target_gpus=8
hard_min_gpus=1
hard_min_cpus=8
planner command: ... --min-gpus 1 --target-gpus 8 ...
```
