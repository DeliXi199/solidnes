# FermiNet Native Spin Beta 10 Iter2000 Submit

Date: 2026-05-26, Asia/Shanghai

## Task

Task `0087`:
`tasks/excited_state_nesvmc/0087_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta10_iter2000/`

This is a single 2000-step pressure test for the native FermiNet PBC
two-state `vmc_overlap` path on carbon diamond Gamma with `spin_penalty=10.0`.
It keeps the task `0086` paper-aligned settings: batch4096, KFAC,
`overlap_penalty=4.0`, `overlap_scale_by=max_gap_std`, bare Hamiltonian energy
output, and `S^2` diagnostics.

## Checks

The local build-only config check passed with:

```text
spin_penalty: 10.0
states: 2
iterations: 2000
batch_size: 4096
kfac_norm_constraint: 0.002
save_path: tasks/excited_state_nesvmc/0087_.../results/checkpoints
```

The Slurm dry-run selected `amdgpu80g/gpu002`, `gpu:4`, 64 CPU cores,
exclusive allocation, and 1 hour walltime.

## Submission

Submitted as Slurm job `129431`.

```text
partition: amdgpu80g
node: gpu002
gres: gpu:4
cpus: 64
walltime: 01:00:00
job name: solidnes-0087-beta10-2000
```

## Expected Result

The useful outcome is either stable reduction of both state `S^2` values toward
zero or a clear failure mode showing that `spin_penalty=10.0` is too strong for
the current optimizer.
