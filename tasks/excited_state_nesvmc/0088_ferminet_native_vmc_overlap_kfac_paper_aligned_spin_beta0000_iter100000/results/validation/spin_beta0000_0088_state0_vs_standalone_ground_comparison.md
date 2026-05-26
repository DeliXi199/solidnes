# Task 0088 State0 vs Standalone Ground-State Comparison

Current task 0088 state0 uses `energy_matrix[:, 0]`, not the scalar `vmc_overlap` loss, so overlap penalty is excluded. The main standalone comparison is task 0044, the x64 single-state FermiNet ground-state 20000-step training run. Task 0046 is the fixed-parameter evaluation of the 0044 checkpoint.

## Summary Table

| dataset | rows | steps | mean (Ha) | std (Ha) | block SE (Ha) | p05-p95 (Ha) | drift last-first decile (Ha) | slope (Ha/1000 steps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0088 excited-state state0 tail30000 | 30000 | 70000-99999 | -75.097157 | 0.128132 | 0.005148 | -75.255450..-74.879661 | 0.046313 | 0.000662 |
| 0088 excited-state state0 tail10000 | 10000 | 90000-99999 | -75.086988 | 0.127184 | 0.008010 | -75.251107..-74.859497 | 0.074780 | 0.003407 |
| 0088 excited-state state0 tail5000 | 5000 | 95000-99999 | -75.081659 | 0.141433 | 0.020601 | -75.253794..-74.802037 | -0.001710 | 0.010206 |
| 0044 standalone ground x64 training tail10000 | 10000 | 10000-19999 | -75.411406 | 0.013792 | 0.000709 | -75.433346..-75.388631 | -0.007216 | -0.000724 |
| 0044 standalone ground x64 training tail5000 | 5000 | 15000-19999 | -75.413151 | 0.012197 | 0.000305 | -75.433101..-75.393326 | -0.001845 | -0.000535 |
| 0046 standalone ground x64 fixed-eval tail1000 | 1000 | 1000-1999 | -75.411863 | 0.012212 | 0.000648 | -75.432210..-75.393461 | -0.002736 | 0.000087 |

## Direct Comparison

- 0088 state0 tail10000 is higher than 0044 standalone ground tail10000 by `0.324418 Ha` (`8.828 eV`).
- 0088 state0 tail30000 is higher than 0044 standalone ground tail10000 by `0.314249 Ha` (`8.551 eV`).
- 0088 state0 tail10000 block SE is `11.3x` the 0044 tail10000 block SE; raw tail std is `9.2x` larger.
- Interpretation: the current excited-state state0 channel is both much higher in energy and much less statistically stable than the earlier standalone ground-state FermiNet calculation.

## Figures

- `spin_beta0000_0088_state0_vs_standalone_ground_comparison.png` / `.svg`
