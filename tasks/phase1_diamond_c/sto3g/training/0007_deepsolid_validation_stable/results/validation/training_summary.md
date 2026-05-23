# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_stable`
Created: `2026-05-21T12:39:56.955507+00:00`

## Training Stats

- Rows: 50
- First step: 0
- Last step: 49
- First energy: -28.2425761797 Ha
- Last energy: -41.5337426643 Ha
- Minimum energy: -49.0733496833 Ha
- Energy delta: -13.2911664846 Ha
- Energy trend: down
- First variance: 99.2860925514
- Last variance: 83.5745132634
- Variance trend: down
- Mean pmove: 0.5425
- Pmove range: [0.3875, 0.7125]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 25
- Tail start step: 25
- Tail energy mean: -36.9712076338 Ha
- Tail energy stderr: 0.919264756714 Ha
- Block count: 5
- Block energy stderr: 1.89549629532 Ha
- Tail variance mean: 118.717038673
- Tail pmove mean: 0.517

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 32.4704540672 Ha
- VMC min minus HF: 24.9308470482 Ha
- VMC tail mean minus HF: 37.0329890977 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
