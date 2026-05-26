# FermiNet Native Spin Penalty Plumbing

Date: 2026-05-25, Asia/Shanghai

## Summary

Implemented the spin-penalty path for SolidNES native FermiNet excited-state
runs.

SolidNES training configs now accept:

```text
spin_penalty: <float>
observables_s2: true|false
```

The adapter maps `spin_penalty` to FermiNet `cfg.optim.spin_energy` and enables
`cfg.observables.s2` automatically when the penalty is positive unless the YAML
overrides it.

## Runtime Fix

The first GPU smoke, Slurm job `129305`, failed because upstream FermiNet's
spin wrapper assumed excited-state local energy returns a non-null auxiliary
energy matrix. The SolidNES PBC `vmc_overlap` path uses state-specific local
energies, where the local-energy function returns a per-state vector and
`aux_data=None`.

The wrapper now handles both cases:

```text
state-specific vector path: E_i -> E_i + beta * diag(S^2)_i
matrix path:                Tr(H) -> Tr(H) + beta * Tr(S^2)
                            H_ij  -> H_ij + beta * S^2_ij
```

## Validation

Source checks:

```text
python -m compileall external/ferminet/ferminet/train.py \
  src/solidnes/backends/ferminet_adapter.py \
  scripts/validation/summarize_ferminet_native_excited_run.py \
  scripts/validation/check_ferminet_native_overlap_loss_alignment.py

JAX_PLATFORMS=cpu python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
```

GPU smoke:

```text
Run: 0080
Task root: tasks/excited_state_nesvmc/0080_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_smoke
Retry job: 129306
State: COMPLETED
Exit code: 0:0
Elapsed: 00:02:02
Node: amdgpu80g/gpu002
GPUs: 4 x A100 80GB
```

Final smoke diagnostics:

```text
rows: 2
final_energy: -22.560371
final_state_energy: [-22.558813, -22.629313]
final_symmetric_overlap_offdiag: 0.0740608
final_s2_matrix: [[2.057880, 0.681531], [-0.214348, 1.416368]]
final_s2_trace: 3.474248
s2_matrix_frames: 2
```

The smoke is a plumbing/runtime check only; it is too short for any physical
energy-gap interpretation.
