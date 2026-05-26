# 0078 FermiNet Native Paper-Aligned 1000-Step Result

Date: 2026-05-25

## Summary

Run `0078` completed the longer paper-aligned native FermiNet PBC
excited-state trajectory.

```text
Job: 129262
Node: amdgpu40g/gpu005
Resources: 4 x A100 40GB, 64 CPU cores, exclusive allocation
Elapsed: 00:03:40
Exit: 0:0
Rows: 1000
Steps: 0 -> 999
```

## Configuration

```text
Experiment:
configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned.yaml

Objective: native FermiNet vmc_overlap
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

## Results

```text
Energy: -22.049490 -> -73.959910
Final EW mean: -73.935610
Final EW variance: 0.003485558
Energy range: [-74.104065, -21.635075]
Tail-50 mean: -73.920053
Tail-100 mean: -73.867186
Tail-200 mean: -73.773711

Mean pmove: 0.664890
Final pmove: 0.533130
Tail-100 pmove from log: 0.539200

Final state energy: [-74.176765, -73.526321]
Final overlap matrix: [[1.0, 0.0186032], [-0.0343843, 1.0]]
Final overlap-gradient scale: [[5.0, 5.0], [5.0, 5.0]]
Final state ordering: [0, 1]
```

The run wrote all expected native excited-state diagnostics:

```text
train_stats.csv
energy_matrix.npy
overlap_matrix.npy
overlap_gradient_scale.npy
overlap_state_ordering.npy
native_ferminet_excited_summary.md
native_ferminet_excited_summary.json
```

## Interpretation

The method-side result is good enough to move beyond smoke validation:

- the native FermiNet excited-state path ran all 1000 KFAC iterations on 4 GPUs;
- all new diagnostics produced 1000 frames;
- the overlap remained small at the end;
- state energies separated by about `0.6504 Ha`;
- the sampler acceptance settled to the same range as the validated ground-state
  FermiNet runs.

The final energy is still above the converged ground-state benchmark
(`-75.4 Ha` scale), as expected for only 1000 training steps from scratch. The
main conclusion from this run is that the paper-aligned native architecture is
stable and fast enough for longer trajectories.

## Artifacts

```text
tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/results/validation/native_ferminet_excited_summary.md
tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/results/validation/native_ferminet_excited_summary.json
tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/logs/slurm/solidnes-0078-native-paper-1000_129262.err
tasks/excited_state_nesvmc/0078_ferminet_native_vmc_overlap_kfac_paper_aligned/logs/slurm/solidnes-0078-native-paper-1000_129262.log
```
