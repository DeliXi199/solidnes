# Task 0086: Native FermiNet Spin Beta Sweep, 2000 Iterations

## Purpose

Run a narrower spin-penalty beta sweep for the paper-aligned native FermiNet
PBC two-state `vmc_overlap` path after the first 1000-step sweep showed that
large beta values were too unstable or too strong.

## Sweep

All variants use:

- Carbon diamond primitive cell at Gamma.
- Native FermiNet PBC excited-state `vmc_overlap` path.
- KFAC optimizer.
- Batch size 4096.
- 2000 iterations.
- Overlap penalty alpha 4.0.
- Bare Hamiltonian energy output for physical gap analysis.
- `S^2` diagnostics enabled.
- Slurm partition `amdgpu80g`.
- One full node: 4 GPUs, 64 CPU cores, exclusive allocation, full node memory.

## Variants

| Variant | Spin beta | Slurm job |
| --- | ---: | --- |
| `runs/beta0000` | 0.000 | 129314 |
| `runs/beta0001` | 0.001 | 129327 |
| `runs/beta0002` | 0.002 | 129328 |
| `runs/beta0005` | 0.005 | 129329 |
| `runs/beta0008` | 0.008 | 129330 |
| `runs/beta0010` | 0.010 | 129331 |
| `runs/beta0012` | 0.012 | 129332 |
| `runs/beta0015` | 0.015 | 129333 |
| `runs/beta0018` | 0.018 | 129334 |
| `runs/beta0020` | 0.020 | 129335 |
| `runs/beta0025` | 0.025 | 129336 |
| `runs/beta0030` | 0.030 | 129337 |

## Submission Notes

The first `beta0000` job was submitted as job 129314 and selected `gpu002`
immediately. The initial tee wrapper failed to open its submit-output file
before the planner created the output directory, but the Slurm submission
itself succeeded.

The remaining jobs were first submitted as 129315--129325, then cancelled before
running because the queued Slurm request inherited the script default
`--mem 64000`. They were re-submitted as 129327--129337 with
`SOLIDNES_GPU_QUEUE_MEMORY_MB=0`, so the final Slurm requests use
`--exclusive` without a memory cap and report `ReqTRES=cpu=64,mem=500G,node=1,gres/gpu=4`.

## Expected Analysis

After all jobs complete, summarize each subrun with
`scripts/validation/summarize_ferminet_native_excited_run.py`, then build a
single comparison table and evolution plot under `results/validation/`.

## Result Summary

All 12 final jobs completed on `amdgpu80g/gpu002` with exit code `0:0` and
wrote 2000 training rows. The analysis artifacts were generated under
`results/validation/`, including the comparison table, per-beta series CSV, and
gap evolution plots.

No tested beta is ready for a 10000-step production run at the current KFAC
settings. The least bad continuation point is `beta=0.008`: final bare gap
`15.936 eV`, tail200 bare gap `16.463 +/- 9.490 eV`, low final symmetric
overlap, and reasonable tail `S^2` diagnostics. `beta=0.002` is the cleanest
low-beta control but gives a lower tail gap, and `beta=0.018` is useful as a
higher-beta control. `beta=0.012` and `beta=0.030` had transient non-finite
diagnostic frames.

Next step: rerun around `beta=0.008` with a less aggressive optimizer and/or
add excited-state pretraining before committing to a long spin-penalty run.

## Last-1000 State Energy Plots

Per-beta plots of the final 1000 training steps for the two physical state
energies were added under
`results/validation/last1000_state_energy/`. The summary index is
`results/validation/last1000_state_energy/spin_beta_sweep_0086_last1000_state_energy_plots.md`,
and the combined 12-panel overview is
`results/validation/last1000_state_energy/spin_beta_sweep_0086_last1000_state_energy_grid.png`.
