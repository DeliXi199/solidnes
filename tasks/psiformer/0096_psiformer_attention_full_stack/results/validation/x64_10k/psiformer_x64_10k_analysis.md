# PsiFormer x64 10k attention comparison

| Variant | Job | Node | s/iter | Final EW mean (Ha) | Tail1000 energy mean (Ha) | Tail1000 energy std | Tail1000 gap (eV) | Slow compile warnings |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upstream/FermiNet x64 | `132575` | `gpu005/amdgpu40g` | 1.232086 | -75.330172 | -75.269677 | 0.180849 | 8.348944 +/- 6.884363 | 2 |
| fused-QKV x64 | `132574` | `gpu002/amdgpu80g` | 1.081922 | -75.301841 | -75.295894 | 0.076742 | 6.881154 +/- 3.120425 | 0 |

Notes:
- Both runs completed 10000 train rows with fp64/x64, two states, no spin penalty, and no S2 observable.
- Speed is not a same-node comparison: fused-QKV ran on gpu002/amdgpu80g, upstream/FermiNet ran on gpu005/amdgpu40g.
- State energies are sorted by instantaneous energy for ground/excited/gap plotting.

Gap stability:

| Variant | Last100 gap (eV) | Last500 gap (eV) | Last1000 gap (eV) | Final gap (eV) |
| --- | ---: | ---: | ---: | ---: |
| upstream/FermiNet x64 | 5.873441 +/- 1.805920 | 6.628265 +/- 3.346492 | 8.348944 +/- 6.884363 | 4.046444 |
| fused-QKV x64 | 6.026690 +/- 2.159662 | 6.418864 +/- 2.388577 | 6.881154 +/- 3.120425 | 4.461400 |

1000-step bin means over final 8000 steps:

| Step range | upstream/FermiNet gap (eV) | fused-QKV gap (eV) |
| --- | ---: | ---: |
| 2000-2999 | 7.8691 +/- 3.3875 | 9.3755 +/- 4.0225 |
| 3000-3999 | 7.4307 +/- 3.2174 | 10.2003 +/- 7.7046 |
| 4000-4999 | 8.1576 +/- 4.1922 | 6.5776 +/- 2.4360 |
| 5000-5999 | 7.3505 +/- 3.1858 | 7.4931 +/- 3.0235 |
| 6000-6999 | 6.9641 +/- 2.6400 | 6.9910 +/- 2.7824 |
| 7000-7999 | 7.3567 +/- 4.0690 | 6.7918 +/- 2.7008 |
| 8000-8999 | 8.2341 +/- 8.7520 | 8.4786 +/- 5.5511 |
| 9000-9999 | 8.3489 +/- 6.8844 | 6.8812 +/- 3.1204 |

Interpretation:
- Fused-QKV has the cleaner scalar training trace over the final 1000 steps:
  lower energy standard deviation and lower EW variance.
- Upstream/FermiNet reaches a lower final EW mean, but its final-1000 energy
  and gap traces are noisier.
- Neither 10k gap trace should be treated as converged. The last-1000 gap
  standard deviations are still several eV, especially for upstream/FermiNet.

Plots:
- `psiformer_x64_10k_iteration_comparison_full.png`
- `psiformer_x64_10k_iteration_comparison_after1000.png`
- `psiformer_x64_10k_iteration_comparison_last1000.png`
- `psiformer_x64_10k_gap_last8000_trailing1000.png`
- `psiformer_x64_10k_gap_last8000_trailing1000_ylim0_15.png`
- `psiformer_x64_10k_gap_last8000_trailing1000_zoom4_12.png`
- `psiformer_x64_10k_gap_last8000_trailing1000_meanonly_zoom4_12.png`
- `psiformer_x64_10k_state_energy_gap_comparison_full.png`
- `psiformer_x64_10k_state_energy_gap_comparison_after1000.png`
- `psiformer_x64_10k_state_energy_gap_comparison_last1000.png`
