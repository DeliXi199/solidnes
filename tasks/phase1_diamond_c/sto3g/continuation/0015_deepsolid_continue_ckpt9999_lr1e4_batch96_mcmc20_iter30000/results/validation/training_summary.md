# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000`
Created: `2026-05-21T16:39:06.216016+00:00`

## Training Stats

- Rows: 20000
- First step: 10000
- Last step: 29999
- First energy: -72.7660984471 Ha
- Last energy: -73.6630777916 Ha
- Minimum energy: -79.0778413727 Ha
- Energy delta: -0.896979344447 Ha
- Energy trend: down
- First variance: 31.93785597
- Last variance: 34.7945466265
- Variance trend: up
- Mean pmove: 0.531274713542
- Pmove range: [0.455729166667, 0.605729166667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 10000
- Tail start step: 20000
- Tail energy mean: -73.9580828434 Ha
- Tail energy stderr: 0.00555941800807 Ha
- Block count: 10
- Block energy stderr: 0.0217050806891 Ha
- Tail variance mean: 29.2206377812
- Tail pmove mean: 0.531996458333

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 0.341118940014 Ha
- VMC min minus HF: -5.0736446411 Ha
- VMC tail mean minus HF: 0.0461138882054 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
