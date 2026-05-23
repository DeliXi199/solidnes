# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_medium`
Created: `2026-05-21T13:35:22.812496+00:00`

## Training Stats

- Rows: 1000
- First step: 0
- Last step: 999
- First energy: -31.3373461704 Ha
- Last energy: -54.2864781349 Ha
- Minimum energy: -81.2861868448 Ha
- Energy delta: -22.9491319645 Ha
- Energy trend: down
- First variance: 85.8857028206
- Last variance: 67.7678927469
- Variance trend: down
- Mean pmove: 0.524675
- Pmove range: [0.3125, 0.6875]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -51.334478675 Ha
- Tail energy stderr: 0.221863026526 Ha
- Block count: 5
- Block energy stderr: 0.542992150354 Ha
- Tail variance mean: 311.780042361
- Tail pmove mean: 0.5252

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 19.7177185967 Ha
- VMC min minus HF: -7.28199011322 Ha
- VMC tail mean minus HF: 22.6697180566 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
