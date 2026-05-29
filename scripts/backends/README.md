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

This folder contains backend-specific run notes and thin wrapper scripts. Add
production launchers only after the backend environment has been confirmed.

For FermiNet/PsiFormer runs launched through `run_ferminet_train.py`,
SolidNES enforces the project long-run checkpoint rule: if
`optim.iterations >= 1000`, the final training step is checkpointed and the
runner fails if the expected final `qmcjax_ckpt_{N-1}.npz` file is missing.
DeepSolid configs built through the adapter set step-based checkpointing for
the same long-run rule when the training config does not override
`checkpoint_every_steps`.

## FermiNet Excited-State Driver

Run the sampler-integrated PBC excited-state driver locally with a cheap
local-energy stand-in:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/backends/run_ferminet_pbc_excited_driver.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_driver_controlled12_walkers4.yaml \
  --local-energy-source cheap --platform cpu
```

Useful optimizer/stability knobs for the same runner:

```bash
--optimizer adam \
--overlap-ewma-decay 0.95 \
--param-share-keys "layers/streams" \
--candidate-check-period 4 \
--max-grad-l2-norm 1000 \
--max-update-l2-norm 1e-4
```

`--optimizer lamb` and `--optimizer kfac` are also available.  The KFAC path
uses `kfac_jax.Optimizer` directly on the external-state paper-tangent loss and
accepts the usual damping/norm controls:

```bash
--optimizer kfac \
--kfac-damping 0.001 \
--kfac-norm-constraint 0.001 \
--kfac-invert-every 1
```

Run the scheduled real PBC local-energy trajectory through the approved FermiNet
GPU submitter:

```bash
TASK=tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4
SOLIDNES_TASK_ROOT="$TASK" \
SOLIDNES_BACKEND_SCRIPT=scripts/backends/run_ferminet_pbc_excited_driver.py \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_driver_controlled12_walkers4.yaml \
SOLIDNES_CONDA_ENV=solidnes-ferminet-jax0101-cuda12 \
SOLIDNES_JOB_NAME=solidnes-0070-nes-drv12 \
bash scripts/slurm/submit_ferminet_gpu_smoke.sh
```
