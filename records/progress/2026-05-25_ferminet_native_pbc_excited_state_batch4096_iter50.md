# FermiNet Native PBC Excited-State Batch4096 Iter50

Date: 2026-05-25

## Summary

Run 0076 is the first native FermiNet PBC excited-state speed/stability baseline
at the same global batch size used by the validated ground-state FermiNet KFAC
runs.

The run used:

- `cfg.system.states = 2`
- `cfg.optim.objective = vmc_overlap`
- native FermiNet KFAC
- native FermiNet multi-device training
- batch size `4096`
- 50 optimizer iterations
- FOLX forward Laplacian

## Task

Task:
`tasks/excited_state_nesvmc/0076_ferminet_native_vmc_overlap_kfac_batch4096_iter50/`

Config:
`configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_batch4096_iter50.yaml`

Train config:
`configs/train/excited_state_ferminet_pbc_native_kfac_batch4096_iter50.yaml`

## Scheduler Result

- Job: `129249`
- State: `COMPLETED`
- Exit code: `0:0`
- Slurm elapsed: `00:03:14`
- Backend log window: `149` seconds
- Node: `amdgpu40g/gpu005`
- Resources: 4 A100 40GB GPUs, 64 CPU cores, exclusive allocation
- JAX devices: `[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]`
- Mid-run GPU sample: all four GPUs had about `36.8 GB` allocated

## Native FermiNet Diagnostics

KFAC registered the native excited-state loss with per-device shape:

```text
NormalMeanNegativeLogProbLoss(mean=...:float32[1024,2], ...)
```

This matches global batch4096 sharded over four devices with two states.

The run completed all 50 rows:

```text
steps: 0 -> 49
energy: -22.054876 -> -24.691084
final_ewmean: -24.117025
final_ewvar: 0.3902106
mean_pmove: 0.9103095788
```

Final native excited-state diagnostics:

```text
final_state_energy: [-25.515295028686523, -23.108867645263672]
final_overlap_matrix: [[1.0, 0.0672965943813324], [0.32805609703063965, 1.0]]
```

## Speed Note

The benchmark summary reports:

```text
elapsed_seconds: 149.000000
seconds_per_step: 2.980000
```

This is a short 50-step run, so the value still includes native excited-state
startup, burn-in, and compilation overhead. It should not be treated as a
steady-state speed number. For comparison, the already validated ground-state
FermiNet batch4096/FOLX KFAC runs had much longer amortization windows, for
example:

- Run 0041: `0.197 s/step` over 2000 steps on one A100 80GB GPU
- Run 0043: `0.1748 s/step` over 10000 steps on two A100 80GB GPUs

The next speed step should therefore be a longer native excited-state run
after the compile path is known to work, not another 50-step short benchmark.

## Outputs

- `results/checkpoints/train_stats.csv`
- `results/checkpoints/energy_matrix.npy`
- `results/checkpoints/overlap_matrix.npy`
- `results/validation/native_ferminet_excited_summary.json`
- `results/validation/native_ferminet_excited_summary.md`
- `results/validation/benchmark_summary.json`
- `results/validation/benchmark_summary.md`
