# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_iter5000_batch96_mcmc12_full`
Created: `2026-05-21T14:21:30.870192+00:00`

## Training Stats

- Rows: 5000
- First step: 0
- Last step: 4999
- First energy: -55.1440831561 Ha
- Last energy: -72.4107097191 Ha
- Minimum energy: -79.3804751803 Ha
- Energy delta: -17.266626563 Ha
- Energy trend: down
- First variance: 234.426918484
- Last variance: 41.4992512729
- Variance trend: down
- Mean pmove: 0.522989583333
- Pmove range: [0.413194444444, 0.663194444444]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 2500
- Tail start step: 2500
- Tail energy mean: -71.2518826225 Ha
- Tail energy stderr: 0.0233135665108 Ha
- Block count: 5
- Block energy stderr: 0.197341611277 Ha
- Tail variance mean: 113.488814629
- Tail pmove mean: 0.521229513889

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 1.59348701246 Ha
- VMC min minus HF: -5.37627844872 Ha
- VMC tail mean minus HF: 2.75231410915 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
