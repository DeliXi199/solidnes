# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_net_batch384_mcmc20_adam_sanity`
Created: `2026-05-22T02:57:10.727890+00:00`

## Training Stats

- Rows: 1000
- First step: 0
- Last step: 999
- First energy: -70.8880404654 Ha
- Last energy: -72.8788682535 Ha
- Minimum energy: -74.7238269243 Ha
- Energy delta: -1.99082778808 Ha
- Energy trend: down
- First variance: 187.655995107
- Last variance: 38.6162605732
- Variance trend: down
- Mean pmove: 0.518435546875
- Pmove range: [0.261848958333, 0.586067708333]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 500
- Tail start step: 500
- Tail energy mean: -72.648879584 Ha
- Tail energy stderr: 0.0270086831285 Ha
- Block count: 5
- Block energy stderr: 0.152337947505 Ha
- Tail variance mean: 71.1787114241
- Tail pmove mean: 0.529111979167

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: 2.09689092567 Ha
- VMC min minus HF: 0.251932254831 Ha
- VMC tail mean minus HF: 2.32687959516 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
