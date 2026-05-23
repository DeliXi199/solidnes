# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_evaluate_ckpt4999_batch96_mcmc20`
Created: `2026-05-21T14:58:33.113143+00:00`

## Training Stats

- Rows: 5000
- First step: 5000
- Last step: 9999
- First energy: -72.6867199339 Ha
- Last energy: -71.317482847 Ha
- Minimum energy: -77.7454452686 Ha
- Energy delta: 1.36923708698 Ha
- Energy trend: up
- First variance: 45.0951996294
- Last variance: 39.5751851678
- Variance trend: down
- Mean pmove: 0.5164384375
- Pmove range: [0.461458333333, 0.594791666667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 5000
- Tail start step: 5000
- Tail energy mean: -72.0376613607 Ha
- Tail energy stderr: 0.0119085343149 Ha
- Block count: 10
- Block energy stderr: 0.0217095985414 Ha
- Tail variance mean: 77.3474584257
- Tail pmove mean: 0.5164384375

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 2.68671388465 Ha
- VMC min minus HF: -3.74124853695 Ha
- VMC tail mean minus HF: 1.96653537094 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
