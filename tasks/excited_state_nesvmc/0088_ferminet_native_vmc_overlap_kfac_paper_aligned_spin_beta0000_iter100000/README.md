# Task 0088: Native FermiNet Beta 0, 100000 Iterations

## Purpose

Run a long beta=0 baseline for the paper-aligned native FermiNet PBC
excited-state `vmc_overlap` path. This disables explicit spin penalty while
keeping `S^2` diagnostics enabled, so the run measures how the excited-state
calculation behaves without spin control over a 100000-step trajectory.

## Configuration

- System: carbon diamond primitive cell at Gamma.
- Backend: native FermiNet PBC excited-state `vmc_overlap`.
- States: 2.
- Batch size: 4096.
- Iterations: 100000.
- Optimizer: KFAC.
- Overlap penalty alpha: 4.0.
- Overlap scaling: `max_gap_std`.
- Spin penalty beta: 0.0.
- `S^2` diagnostics: enabled.
- Pretraining: disabled.

## Files

- Experiment config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_beta0000_iter100000.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_native_kfac_paper_aligned_spin_beta0000_iter100000.yaml`

## Notes

This is a long baseline, not a spin-control fix. It should be interpreted
against task `0086` beta=0 and the task `0087` beta=10 pressure-test failure
mode.

## Submission

Build-only config check passed with:

```text
states: 2
spin_penalty: 0.0
s2_observable: True
iterations: 100000
batch_size: 4096
save_path: tasks/excited_state_nesvmc/0088_.../results/checkpoints
```

Slurm dry-run selected `amdgpu80g/gpu002`, `gpu:4`, 64 CPU cores, exclusive
allocation, and `08:00:00` walltime. The job was submitted as:

```text
job_id: 129450
job_name: solidnes-0088-beta0000-100k
state_at_submit_check: RUNNING
node: gpu002
```

## Result

Slurm job `129450` completed on `amdgpu80g/gpu002` with exit code `0:0` in
`03:46:24`. The run wrote all 100000 rows and expected matrix diagnostics.

Key result:

- Final scalar energy: `-75.037605 Ha`.
- Final state energies: `[-75.096855, -74.919098] Ha`.
- Final gap: `4.837 eV`; tail200 gap median: `9.205 eV`.
- Final `S^2` diagonal / trace: `[1.4585, 81.4612] / 82.9197`.
- `S^2` diagnostics contain 35 non-finite frames, including 10 trace
  non-finite frames in the last 10000 steps.
- Last 10000 finite frames include 139 frames with `|S^2 trace| > 10` and
  34 frames with `|S^2 trace| > 50`.

Interpretation: the beta=0 long baseline completed cleanly as a Slurm job and
the energy trajectory is usable as a no-spin-penalty reference, but it confirms
that the excited-state spin is not controlled without an explicit method-side
fix.

Analysis artifacts:

- `results/validation/native_ferminet_excited_summary.md`
- `results/validation/spin_beta0000_0088_analysis.md`
- `results/validation/spin_beta0000_0088_analysis.json`
- `results/validation/spin_beta0000_0088_evolution_overview.png`
- `results/validation/spin_beta0000_0088_last10000_state_energy.png`
- `results/validation/spin_beta0000_0088_last10000_gap_s2.png`
