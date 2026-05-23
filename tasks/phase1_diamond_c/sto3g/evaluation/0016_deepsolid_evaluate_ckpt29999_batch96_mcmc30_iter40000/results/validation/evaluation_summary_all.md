# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000`
Created: `2026-05-21T17:04:37.700919+00:00`

## Training Stats

- Rows: 10000
- First step: 30000
- Last step: 39999
- First energy: -73.664151743 Ha
- Last energy: -73.1775560067 Ha
- Minimum energy: -77.5493119875 Ha
- Energy delta: 0.486595736288 Ha
- Energy trend: up
- First variance: 43.3781356656
- Last variance: 14.5489917202
- Variance trend: down
- Mean pmove: 0.537692395833
- Pmove range: [0.471527777778, 0.590277777778]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 10000
- Tail start step: 30000
- Tail energy mean: -74.0217682184 Ha
- Tail energy stderr: 0.0049939556062 Ha
- Block count: 10
- Block energy stderr: 0.00693292390606 Ha
- Tail variance mean: 24.3897387939
- Tail pmove mean: 0.537692395833

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 0.826640724935 Ha
- VMC min minus HF: -3.54511525594 Ha
- VMC tail mean minus HF: -0.017571486804 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
