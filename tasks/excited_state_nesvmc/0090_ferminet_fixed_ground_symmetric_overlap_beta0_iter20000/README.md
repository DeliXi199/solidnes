# 0090 FermiNet Fixed-Ground Symmetric Overlap Beta0 Iter20000

## Purpose

Submit a beta=0/no-spin fixed-ground excited-state retry after replacing the
one-sided fixed-reference overlap with a symmetric estimator:

```text
S_ge^2 = E_exc[psi_ground / psi_excited] * E_ground[psi_excited / psi_ground]
```

The ground-state checkpoint remains fixed. Only the excited-state parameters are
optimized.

## Configuration

- Experiment: `configs/experiment/diamond_c_ferminet_pbc_gamma_fixed_ground_symmetric_overlap_beta0_iter20000.yaml`
- Train config: `configs/train/excited_state_ferminet_pbc_fixed_ground_symmetric_overlap_beta0_iter20000.yaml`
- Fixed ground checkpoint: `tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints/qmcjax_ckpt_018349.npz`
- Iterations: `20000`
- Batch size: `4096`
- Optimizer: KFAC/FOLX
- Spin penalty: `0.0`
- S2 observable: disabled

## Requested Slurm Resources

- Partition: `gpu80gllm`
- GPUs: `4`
- CPU cores: `64`
- Allocation: exclusive single node
- Walltime: `12:00:00`

## Submission

- Direct `gpu80gllm` submission was rejected by Slurm policy because the
  current user account is not in `gpu80gllm`'s `AllowAccounts` list.
- Invalid `gpu80gllm` attempt: job `130202`, cancelled after it entered
  `PD (PartitionConfig)`.
- Intermediate `amdgpu80g` replacement: job `130203`, cancelled while pending
  after the user requested `amdgpu40g`.
- First `amdgpu40g` attempt: job `130207`, failed during burn-in with a JAX
  donated-buffer aliasing error; fixed by copying fixed-reference walker
  buffers before donation.
- Effective job: `130209`
- Effective partition: `amdgpu40g`
- Effective node: `gpu007`
- Effective resources: `gpu:4`, `64` CPU cores, exclusive single node,
  `--mem=0`, walltime `12:00:00`.
- State after submission: `RUNNING`.
