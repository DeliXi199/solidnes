# Backend Scripts

Utilities for checking and adapting external NQS backends.

## DeepSolid

Check the Python environment:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/check_deepsolid_environment.py
```

Build the first carbon-diamond config smoke:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/build_deepsolid_config.py --create-output-dirs
```

Build the lower-cost runtime smoke config:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/build_deepsolid_config.py \
  configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml \
  --create-output-dirs
```

Run the lower-cost runtime process smoke:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/run_deepsolid_process_smoke.py
```

Probe reusable DeepSolid adapter objects without entering the training loop:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/backends/probe_deepsolid_adapter_objects.py \
  configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml
```

Run the one-step Adam GPU smoke:

```bash
SOLIDNES_GPU_PARTITIONS=h200,amdgpu80g,amdgpu40g,intelgpu80g,h20 \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_deepsolid_one_step_adam_smoke.yaml \
SOLIDNES_JOB_NAME=NNNN_deepsolid_one_step_adam_smoke \
SOLIDNES_TASK_ROOT=tasks/phase1_diamond_c/sto3g/smoke/NNNN_deepsolid_one_step_adam_smoke \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

Run the adapter-object probe on GPU:

```bash
SOLIDNES_GPU_PARTITIONS=h200,amdgpu80g,amdgpu40g,intelgpu80g,h20 \
SOLIDNES_BACKEND_SCRIPT=scripts/backends/probe_deepsolid_adapter_objects.py \
SOLIDNES_JOB_NAME=NNNN_deepsolid_adapter_probe \
SOLIDNES_TASK_ROOT=tasks/phase1_diamond_c/sto3g/smoke/NNNN_deepsolid_adapter_probe \
  bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
```

This folder will contain backend-specific run notes and thin wrapper scripts.

Do not add production launchers before the backend environment has been
confirmed.
