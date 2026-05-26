# FermiNet Native Spin Beta 0 Iter100000 Submit

Date: 2026-05-26

Submitted task `0088`, a requested long beta=0 baseline for the native
FermiNet PBC excited-state path:

```text
task: tasks/excited_state_nesvmc/0088_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta0000_iter100000
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_beta0000_iter100000.yaml
train_config: configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_spin_beta0000_iter100000.yaml
job_id: 129450
job_name: solidnes-0088-beta0000-100k
state_at_submit_check: RUNNING
node: gpu002
partition: amdgpu80g
resources: gpu:4, cpus_per_task=64, exclusive
walltime: 08:00:00
```

Configuration:

- Native FermiNet PBC excited-state `vmc_overlap`.
- Carbon diamond primitive cell at Gamma.
- 2 states, batch4096, KFAC.
- 100000 iterations.
- Overlap penalty alpha `4.0`.
- Overlap scaling `max_gap_std`.
- `spin_penalty=0.0`.
- `S^2` diagnostics enabled.
- Pretraining disabled.

Validation before submission:

- Build-only FermiNet adapter check passed.
- Slurm dry-run selected `amdgpu80g/gpu002` with 4 A100 80GB GPUs, 64 CPU
  cores, exclusive allocation, and 8 hour walltime.

Expected primary outputs:

- `results/checkpoints/train_stats.csv`
- `results/checkpoints/energy_matrix.npy`
- `results/checkpoints/overlap_matrix.npy`
- `results/checkpoints/overlap_symmetric_matrix.npy`
- `results/checkpoints/overlap_penalty_matrix.npy`
- `results/checkpoints/overlap_gradient_scale.npy`
- `results/checkpoints/overlap_state_ordering.npy`
- `results/checkpoints/overlap_scale_energy_ewm.npy`
- `results/checkpoints/overlap_scale_std_ewm.npy`
- `results/checkpoints/s2_matrix.npy`

Purpose: measure the uncontrolled long-trajectory spin behavior of the
paper-aligned excited-state calculation with explicit spin penalty disabled.
