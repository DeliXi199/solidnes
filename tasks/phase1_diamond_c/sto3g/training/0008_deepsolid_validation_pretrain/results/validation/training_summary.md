# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_pretrain`
Created: `2026-05-21T12:55:51.015839+00:00`

## Training Stats

- Rows: 200
- First step: 0
- Last step: 199
- First energy: -32.7957561564 Ha
- Last energy: -44.230042186 Ha
- Minimum energy: -56.8465340966 Ha
- Energy delta: -11.4342860296 Ha
- Energy trend: down
- First variance: 339.214008486
- Last variance: 91.501324336
- Variance trend: down
- Mean pmove: 0.5255
- Pmove range: [0.3625, 0.6875]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 100
- Tail start step: 100
- Tail energy mean: -42.9822478613 Ha
- Tail energy stderr: 0.322200822775 Ha
- Block count: 5
- Block energy stderr: 0.510318151229 Ha
- Tail variance mean: 247.510464948
- Tail pmove mean: 0.5195

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 29.7741545456 Ha
- VMC min minus HF: 17.157662635 Ha
- VMC tail mean minus HF: 31.0219488703 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
