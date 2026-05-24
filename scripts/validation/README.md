# Validation Scripts

Repeatable checks for deciding whether a backend run is merely executable or
also numerically credible.

No-compute excited-state scaffold checks:

```bash
PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py
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
