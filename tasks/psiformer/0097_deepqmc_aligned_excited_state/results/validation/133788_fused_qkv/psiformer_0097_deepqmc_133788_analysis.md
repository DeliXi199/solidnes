# 0097 DeepQMC-Aligned PsiFormer 133788 Analysis

Task: `0097_deepqmc_aligned_excited_state`
Run: `fullnode_anygpu_fused_qkv_x64_deepqmc_b4096_i10000_levmap128_jaxattn`
Checkpoint dir: `tasks/psiformer/0097_deepqmc_aligned_excited_state/runs/fullnode_anygpu_fused_qkv_x64_deepqmc_b4096_i10000_levmap128_jaxattn/results/checkpoints`

## Completion

| Item | Value |
| --- | ---: |
| Rows | 10000 |
| Steps | 0 -> 9999 |
| Elapsed seconds | 22454.4784 |
| Seconds / iteration | 2.24544784 |
| Final checkpoint exists | True |

## Scalar Training Stats

| Metric | Final | Tail100 mean | Tail500 mean | Tail1000 mean |
| --- | ---: | ---: | ---: | ---: |
| Energy (Ha) | -75.3255139 | -75.310787 | -75.3137798 | -75.3098049 |
| EW mean (Ha) | -75.3133738 | -75.3102946 | -75.3136691 | -75.3096894 |
| EW variance | 0.00072363764 | 0.00110066841 | 0.000982751507 | 0.0010532541 |
| Pmove | 0.532189941 | 0.535025513 | 0.535195044 | 0.535175073 |

## State Energies And Gap

| Metric | Value |
| --- | ---: |
| Final state energies (Ha) | `[-75.44005804560628, -75.21097508514882]` |
| Tail100 state-energy mean (Ha) | `[-75.39877843818141, -75.22343831939948]` |
| Tail1000 state-energy mean (Ha) | `[-75.39886920528036, -75.221403910383]` |
| Final fixed gap (Ha) | 0.22908296 |
| Final fixed gap (eV) | 6.23366492 |
| Tail100 fixed gap (eV) | 4.7712477 |
| Tail500 fixed gap (eV) | 4.83902095 |
| Tail1000 fixed gap (eV) | 4.82907668 |
| Final energy-sorted gap (eV) | 6.23366492 |
| Tail1000 energy-sorted gap (eV) | 4.83788923 |
| Fixed-gap negative rows | 1068 |
| Fixed-gap negative rows, last1000 | 7 |
| State ordering final | `[0, 1]` |
| State-ordering swaps | 0 |

## Overlap Diagnostics

| Metric | Value |
| --- | ---: |
| Final overlap matrix | `[[1.0, -0.0007700125107531104], [-0.0008591709621210159, 1.0]]` |
| Final symmetric overlap matrix | `[[1.0, -0.000813371003730136], [-0.000813371003730136, 1.0]]` |
| Final overlap penalty matrix | `[[1.0, 6.61572389708969e-07], [6.61572389708969e-07, 1.0]]` |
| Final gradient scale | `[[3.0181145668029785, 3.0181145668029785], [2.9483859539031982, 2.9483859539031982]]` |
| Final symmetric overlap 01 | -0.000813371004 |
| Tail1000 abs symmetric overlap 01 mean | 0.00529441474 |
| Tail1000 abs symmetric overlap 01 max | 0.033252165 |
| Final overlap penalty 01 | 6.6157239e-07 |

## Artifacts

- Timeseries CSV: `psiformer_0097_deepqmc_133788_timeseries.csv`
- Summary JSON: `psiformer_0097_deepqmc_133788_summary.json`
- Scalar evolution plots: `psiformer_0097_deepqmc_133788_iteration_evolution_*.png` / `.svg`
- State energy and gap plots: `psiformer_0097_deepqmc_133788_state_energy_gap_*.png` / `.svg`
- Overlap plots: `psiformer_0097_deepqmc_133788_overlap_evolution_*.png` / `.svg`
- Ground/excited energy after step 2000 with 1000-step mean: `psiformer_0097_deepqmc_133788_ground_excited_after2000_roll1000.png` / `.svg`
- Gap after step 2000 with 1000-step mean: `psiformer_0097_deepqmc_133788_gap_after2000_roll1000.png` / `.svg`

## Readout

The run completed the requested 10000 iterations and wrote the final checkpoint. The DeepQMC-aligned independent-state route keeps the fixed state ordering `[0, 1]` throughout the run and drives the final symmetric off-diagonal overlap close to zero. The final single-step gap is larger than the tail means, so tail windows are a better read of the late-time behavior than the last row alone.
