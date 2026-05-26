# Task 0086 Last-1000 State Energy Plots

Each per-beta plot shows the final 1000 training steps for the two physical state energies. Raw values are drawn faintly and 50-step rolling means are drawn as the main curves.

`beta=0.000` uses `energy_matrix.npy` because no spin penalty was applied. All other variants use `bare_energy_matrix.npy`, so the plotted values are physical Hamiltonian energies rather than spin-penalized training energies.

Combined overview: `spin_beta_sweep_0086_last1000_state_energy_grid.png` and `spin_beta_sweep_0086_last1000_state_energy_grid.svg`.

| variant | beta | source | tail steps | non-finite tail frames | final state0 Ha | final state1 Ha | tail1000 mean state0 Ha | tail1000 mean state1 Ha | plot |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| beta0000 | 0.000 | `energy_matrix.npy (beta=0 physical/training)` | 1000--1999 | 0 | -74.434067 | -73.918396 | -74.337341 | -73.655709 | `spin_beta_0086_beta0000_last1000_state_energy.png` |
| beta0001 | 0.001 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.362762 | -73.459061 | -74.405689 | -73.600960 | `spin_beta_0086_beta0001_last1000_state_energy.png` |
| beta0002 | 0.002 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.553299 | -74.028877 | -74.370926 | -73.806673 | `spin_beta_0086_beta0002_last1000_state_energy.png` |
| beta0005 | 0.005 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.638184 | -74.450508 | -74.544956 | -74.201996 | `spin_beta_0086_beta0005_last1000_state_energy.png` |
| beta0008 | 0.008 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.722702 | -74.137070 | -74.489666 | -73.856236 | `spin_beta_0086_beta0008_last1000_state_energy.png` |
| beta0010 | 0.010 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.583374 | -74.433624 | -74.530851 | -74.113342 | `spin_beta_0086_beta0010_last1000_state_energy.png` |
| beta0012 | 0.012 | `bare_energy_matrix.npy` | 1000--1999 | 2 | -74.516960 | -74.059853 | -74.454817 | -73.965088 | `spin_beta_0086_beta0012_last1000_state_energy.png` |
| beta0015 | 0.015 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.441322 | -74.378052 | -74.333793 | -73.727631 | `spin_beta_0086_beta0015_last1000_state_energy.png` |
| beta0018 | 0.018 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.516563 | -73.983681 | -74.435745 | -73.688496 | `spin_beta_0086_beta0018_last1000_state_energy.png` |
| beta0020 | 0.020 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.712067 | -74.401642 | -74.493903 | -73.853115 | `spin_beta_0086_beta0020_last1000_state_energy.png` |
| beta0025 | 0.025 | `bare_energy_matrix.npy` | 1000--1999 | 0 | -74.743774 | -74.230324 | -74.382953 | -73.663681 | `spin_beta_0086_beta0025_last1000_state_energy.png` |
| beta0030 | 0.030 | `bare_energy_matrix.npy` | 1000--1999 | 1 | -74.440575 | -73.994629 | -74.299815 | -73.583781 | `spin_beta_0086_beta0030_last1000_state_energy.png` |
