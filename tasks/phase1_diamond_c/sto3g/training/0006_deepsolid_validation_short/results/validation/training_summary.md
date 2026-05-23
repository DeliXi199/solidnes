# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_short`
Created: `2026-05-21T12:27:36.020401+00:00`

## Training Stats

- Rows: 20
- First step: 0
- Last step: 19
- First energy: -48.41835816 Ha
- Last energy: -27.2161501781 Ha
- Minimum energy: -48.41835816 Ha
- Energy delta: 21.2022079819 Ha
- Energy trend: up
- First variance: 3315.267813
- Last variance: 275.944493004
- Variance trend: down
- Mean pmove: 0.54625
- Pmove range: [0.375, 0.75]
- Finite checks: True

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 46.7880465535 Ha
- VMC min minus HF: 25.5858385716 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
