# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000`
Created: `2026-05-23T09:00:54.362795+00:00`

## Training Stats

- Rows: 10000
- First step: 1000
- Last step: 10999
- First energy: -75.3626140667 Ha
- Last energy: -75.4010653122 Ha
- Minimum energy: -75.4945589522 Ha
- Energy delta: -0.0384512455275 Ha
- Energy trend: down
- First variance: 3.85079205141
- Last variance: 0.597879938652
- Variance trend: down
- Mean pmove: 0.539793599854
- Pmove range: [0.531127929688, 0.57939453125]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 2000
- Tail start step: 9000
- Tail energy mean: -75.416127997 Ha
- Tail energy stderr: 0.000288488725098 Ha
- Block count: 5
- Block energy stderr: 0.000645317369747 Ha
- Tail variance mean: 0.698087048561
- Tail pmove mean: 0.539674987793

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: -0.425306133059 Ha
- VMC min minus HF: -0.518799773025 Ha
- VMC tail mean minus HF: -0.440368817884 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
