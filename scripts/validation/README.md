# Validation Scripts

Repeatable checks for deciding whether a backend run is merely executable or
also numerically credible.

No-compute excited-state scaffold checks:

```bash
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
```

FermiNet/JAX build-only excited-state adapter check:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_excited_adapter_build.py
```

FermiNet/JAX build-only penalty-term path check:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_terms.py
```

FermiNet/JAX build-only penalty gradient-step check:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_grad_step.py
```

FermiNet/JAX build-only multi-step cheap optimization smoke through the
reusable fixed-sample paper-tangent training loop. The smoke now accepts
`--optimizer adam|lamb|sgd|kfac`, `--overlap-ewma-decay`,
`--param-share-keys`, `--candidate-check-period`, gradient/update norm caps,
and KFAC damping/norm controls:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py \
  --optimizer kfac --overlap-ewma-decay 0.95 \
  --param-share-keys layers/streams --candidate-check-period 2 \
  --kfac-damping 0.001 --kfac-norm-constraint 0.001
```

FermiNet/JAX build-only sampler-integrated driver smoke with checkpoint
roundtrip. The script reads driver defaults such as `local_energy_source`,
`walkers_per_state`, `iterations`, and sampler settings from the experiment YAML
`diagnostics` section unless command line arguments or `SOLIDNES_NES_*`
environment variables override them.

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_driver_smoke.py
```

Native FermiNet paper-aligned overlap-loss helper check:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env JAX_PLATFORMS=cpu PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
```

PsiFormer attention implementation benchmark:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env JAX_PLATFORMS=cpu PYTHONPATH=external/ferminet:src \
  python scripts/validation/benchmark_psiformer_attention.py \
  --walkers 64 --warmup 2 --repeats 8
```

Use `--platform cuda` for scheduled GPU timing checks in the CUDA FermiNet
environment.

PsiFormer full-node comparison plots after 0096 jobs finish:

```bash
python scripts/validation/plot_psiformer_fullnode_attention_iterations.py
python scripts/validation/plot_psiformer_fullnode_state_gap.py
```

FermiNet/JAX scheduled real PBC local-energy smoke:

```bash
TASK=tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke
SOLIDNES_TASK_ROOT="$TASK" \
SOLIDNES_PLAN_JSON="$TASK/outputs/slurm_plans/fullnode_plan.json" \
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_smoke.py \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_real_local_energy_smoke.yaml \
SOLIDNES_CONDA_ENV=solidnes-ferminet-jax0101-cuda12 \
SOLIDNES_JOB_NAME=solidnes-nes-real-le-full \
SOLIDNES_TIME_LIMIT=00:20:00 \
SOLIDNES_GPU_PARTITIONS=intelgpu80g \
SOLIDNES_GPU_ALLOWED_COUNTS=2 \
SOLIDNES_GPU_TARGET_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_CPUS=96 \
SOLIDNES_GPU_QUEUE_MIN_GPUS=2 \
SOLIDNES_GPU_QUEUE_MIN_CPUS=96 \
SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE=1 \
bash scripts/slurm/submit_ferminet_gpu_smoke.sh
```

FermiNet/JAX scheduled real PBC local-energy multi-step training-loop smoke.
The script reads defaults such as `walkers_per_state`, `steps`, `seed`, and
`learning_rate` from the experiment YAML `diagnostics` section unless command
line arguments or `SOLIDNES_NES_*` environment variables override them.

```bash
TASK=tasks/excited_state_nesvmc/0067_ferminet_pbc_paper_tangent_training_smoke_walkers2
SOLIDNES_TASK_ROOT="$TASK" \
SOLIDNES_PLAN_JSON="$TASK/outputs/slurm_plans/plan.json" \
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml \
SOLIDNES_CONDA_ENV=solidnes-ferminet-jax0101-cuda12 \
SOLIDNES_JOB_NAME=solidnes-0067-nes-pt-w2 \
SOLIDNES_TIME_LIMIT=00:30:00 \
SOLIDNES_GPU_PARTITIONS=intelgpu80g \
SOLIDNES_GPU_ALLOWED_COUNTS=2 \
SOLIDNES_GPU_TARGET_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_CPUS=96 \
SOLIDNES_GPU_QUEUE_MIN_GPUS=2 \
SOLIDNES_GPU_QUEUE_MIN_CPUS=96 \
SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE=1 \
bash scripts/slurm/submit_ferminet_gpu_smoke.sh
```

FermiNet/JAX scheduled sampler-integrated real PBC local-energy driver smoke:

```bash
TASK=tasks/excited_state_nesvmc/0068_ferminet_pbc_driver_real_local_energy_smoke
SOLIDNES_TASK_ROOT="$TASK" \
SOLIDNES_PLAN_JSON="$TASK/outputs/slurm_plans/plan.json" \
SOLIDNES_BACKEND_SCRIPT=scripts/validation/check_ferminet_pbc_driver_smoke.py \
SOLIDNES_EXPERIMENT=configs/experiment/diamond_c_ferminet_pbc_gamma_driver_real_local_energy_smoke.yaml \
SOLIDNES_CONDA_ENV=solidnes-ferminet-jax0101-cuda12 \
SOLIDNES_JOB_NAME=solidnes-0068-nes-driver \
SOLIDNES_TIME_LIMIT=00:30:00 \
SOLIDNES_GPU_PARTITIONS=intelgpu80g \
SOLIDNES_GPU_ALLOWED_COUNTS=2 \
SOLIDNES_GPU_TARGET_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_GPUS=2 \
SOLIDNES_GPU_HARD_MIN_CPUS=96 \
SOLIDNES_GPU_QUEUE_MIN_GPUS=2 \
SOLIDNES_GPU_QUEUE_MIN_CPUS=96 \
SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE=1 \
bash scripts/slurm/submit_ferminet_gpu_smoke.sh
```

Summarize a FermiNet PBC excited-state driver trajectory:

```bash
python scripts/validation/summarize_ferminet_pbc_driver_run.py \
  tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4/results/validation/ferminet_pbc_driver_run_summary.json
```

Carbon diamond HF reference:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/validation/compute_pyscf_pbc_reference.py \
  configs/experiment/diamond_c_deepsolid_validation_pretrain.yaml
```

Carbon diamond cc-pVDZ HF reference for paper alignment:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/validation/compute_pyscf_pbc_reference.py \
  configs/experiment/diamond_c_deepsolid_ccpvdz_hf_reference.yaml
```

Training summary:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/validation/summarize_deepsolid_validation.py \
  configs/experiment/diamond_c_deepsolid_validation_medium.yaml
```

FermiNet benchmark summary:

```bash
python scripts/validation/summarize_ferminet_benchmark.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100.yaml \
  --job-id 127830 \
  --log tasks/phase1_diamond_c/pbc_gamma/training/0033_ferminet_adam_short100/logs/slurm/solidnes-ferminet-short100_127830.log \
  --err tasks/phase1_diamond_c/pbc_gamma/training/0033_ferminet_adam_short100/logs/slurm/solidnes-ferminet-short100_127830.err \
  --plan-json tasks/phase1_diamond_c/pbc_gamma/training/0033_ferminet_adam_short100/outputs/slurm_plans/ferminet_adam_short100_plan.json
```

Current result:

```text
The validation harness works, but the carbon sto-3g DeepSolid runs are not yet
converged or accurate relative to the same-cell HF reference.
HF-target pretraining and the medium model both improve the tail mean, and the
medium 1000-step continuation reaches a tail mean of -51.3345 Ha.
The remaining tail gap to HF is still +22.6697 Ha, so this is optimization
progress rather than an accuracy claim.
```
