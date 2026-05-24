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

FermiNet/JAX build-only multi-step cheap optimization smoke:

```bash
conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py
```

FermiNet/JAX scheduled real PBC local-energy smoke:

```bash
SOLIDNES_TASK_ROOT=tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke \
SOLIDNES_PLAN_JSON=tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/outputs/slurm_plans/fullnode_plan.json \
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
bash scripts/slurm/submit_ferminet_gpu_smoke.sh
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
