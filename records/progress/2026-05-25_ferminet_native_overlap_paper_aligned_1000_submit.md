# 0078 FermiNet Native Paper-Aligned 1000-Step Submission

Date: 2026-05-25

## Purpose

Run a longer native FermiNet PBC excited-state trajectory after the 0077
paper-aligned smoke, so startup and compilation overhead are amortized before
steady-state speed and stability are compared against the ground-state
FermiNet baseline.

## Configuration

```text
Experiment:
configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned.yaml

Train config:
configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned.yaml

Backend: native FermiNet
Objective: vmc_overlap
Optimizer: KFAC
States: 2
Batch size: 4096
Iterations: 1000
Overlap penalty alpha: 4.0
Overlap scale: max_gap_std
Overlap min/max scale: 0.001 / 5.0
Overlap clip width: 10.0
State ordering: energy
Local-energy clipping: median-centered
```

## Submission

```text
Run: 0078
Job: 129262
Partition/node: amdgpu40g/gpu005
Resources: 4 x A100 40GB, 64 CPU cores, exclusive allocation
Walltime: 01:00:00
Task root:
tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/
```

The dry-run plan was forced to `amdgpu40g` by blocking `amdgpu80g` and
selected `gpu005`. The submitted job entered `RUNNING`; startup logs confirmed
JAX 0.10.1 on four GPU devices.

## Expected Outputs

```text
train_stats.csv
energy_matrix.npy
overlap_matrix.npy
overlap_gradient_scale.npy
overlap_state_ordering.npy
native_ferminet_excited_summary.md
native_ferminet_excited_summary.json
```

Final validation should summarize the run with:

```bash
python scripts/validation/summarize_ferminet_native_excited_run.py \
  tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/results/checkpoints \
  --experiment configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned.yaml \
  --job-id 129262
```
