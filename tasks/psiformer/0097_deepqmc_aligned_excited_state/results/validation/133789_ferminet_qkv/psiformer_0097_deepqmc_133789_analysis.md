# 0097 DeepQMC-Aligned PsiFormer 133789 Analysis

Task: `0097_deepqmc_aligned_excited_state`
Run: `fullnode_anygpu_ferminet_x64_deepqmc_b4096_i10000_levmap128_jaxattn`
Checkpoint dir: `tasks/psiformer/0097_deepqmc_aligned_excited_state/runs/fullnode_anygpu_ferminet_x64_deepqmc_b4096_i10000_levmap128_jaxattn/results/checkpoints`

## Completion

| Item | Value |
| --- | ---: |
| Rows | 10000 |
| Steps | 0 -> 9999 |
| Elapsed seconds | 22016.8048 |
| Seconds / iteration | 2.20168048 |
| Final checkpoint exists | True |

## Scalar Training Stats

| Metric | Final | Tail100 mean | Tail500 mean | Tail1000 mean |
| --- | ---: | ---: | ---: | ---: |
| Energy (Ha) | -75.3391858 | -75.3046724 | -75.3055293 | -75.3049075 |
| EW mean (Ha) | -75.3060269 | -75.3049271 | -75.3052586 | -75.3048096 |
| EW variance | 0.0013767541 | 0.000815313956 | 0.000905325065 | 0.000953874312 |
| Pmove | 0.535510254 | 0.535043945 | 0.535099316 | 0.535098633 |

## State Energies And Gap

| Metric | Value |
| --- | ---: |
| Final state energies (Ha) | `[-75.4491703411549, -75.22920118689542]` |
| Tail100 state-energy mean (Ha) | `[-75.40582092534746, -75.2040701998989]` |
| Tail1000 state-energy mean (Ha) | `[-75.4017380831894, -75.20893115292758]` |
| Final fixed gap (Ha) | 0.219969154 |
| Final fixed gap (eV) | 5.98566562 |
| Tail100 fixed gap (eV) | 5.48991692 |
| Tail500 fixed gap (eV) | 5.25120426 |
| Tail1000 fixed gap (eV) | 5.24654385 |
| Final energy-sorted gap (eV) | 5.98566562 |
| Tail1000 energy-sorted gap (eV) | 5.25129153 |
| Fixed-gap negative rows | 1439 |
| Fixed-gap negative rows, last1000 | 3 |
| State ordering final | `[0, 1]` |
| State-ordering swaps | 0 |

## Overlap Diagnostics

| Metric | Value |
| --- | ---: |
| Final overlap matrix | `[[1.0, 0.019137041500670793], [-0.011157305403096894, 1.0]]` |
| Final symmetric overlap matrix | `[[1.0, 0.0], [-0.0, 1.0]]` |
| Final overlap penalty matrix | `[[1.0, 0.0], [0.0, 1.0]]` |
| Final gradient scale | `[[3.0192534923553467, 3.0192534923553467], [2.9963645935058594, 2.9963645935058594]]` |
| Final symmetric overlap 01 | 0 |
| Tail1000 abs symmetric overlap 01 mean | 0.00606314669 |
| Tail1000 abs symmetric overlap 01 max | 0.0384394431 |
| Final overlap penalty 01 | 0 |

## Artifacts

- Timeseries CSV: `psiformer_0097_deepqmc_133789_timeseries.csv`
- Summary JSON: `psiformer_0097_deepqmc_133789_summary.json`
- Scalar evolution plots: `psiformer_0097_deepqmc_133789_iteration_evolution_*.png` / `.svg`
- State energy and gap plots: `psiformer_0097_deepqmc_133789_state_energy_gap_*.png` / `.svg`
- Overlap plots: `psiformer_0097_deepqmc_133789_overlap_evolution_*.png` / `.svg`
- Ground/excited energy after step 2000 with 1000-step mean: `psiformer_0097_deepqmc_133789_ground_excited_after2000_roll1000.png` / `.svg`
- Gap after step 2000 with 1000-step mean: `psiformer_0097_deepqmc_133789_gap_after2000_roll1000.png` / `.svg`

## Readout

The run completed the requested 10000 iterations and wrote the final checkpoint. The DeepQMC-aligned independent-state route keeps the fixed state ordering `[0, 1]` throughout the run and drives the final symmetric off-diagonal overlap close to zero. The final single-step gap is larger than the tail means, so tail windows are a better read of the late-time behavior than the last row alone.
