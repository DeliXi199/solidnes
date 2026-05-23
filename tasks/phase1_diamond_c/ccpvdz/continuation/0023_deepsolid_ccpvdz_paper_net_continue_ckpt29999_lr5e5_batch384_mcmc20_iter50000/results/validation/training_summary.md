# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt29999_lr5e5_batch384_mcmc20_iter50000`
Created: `2026-05-22T08:17:41.945141+00:00`

## Training Stats

- Rows: 20000
- First step: 30000
- Last step: 49999
- First energy: -74.8392740274 Ha
- Last energy: -75.2077802431 Ha
- Minimum energy: -78.5235817156 Ha
- Energy delta: -0.368506215624 Ha
- Energy trend: down
- First variance: 10.2293105497
- Last variance: 6.7055957245
- Variance trend: down
- Mean pmove: 0.539403411458
- Pmove range: [0.487890625, 0.574088541667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 10000
- Tail start step: 40000
- Tail energy mean: -75.0909943902 Ha
- Tail energy stderr: 0.00172605570769 Ha
- Block count: 5
- Block energy stderr: 0.00884642293369 Ha
- Tail variance mean: 11.4795113073
- Tail pmove mean: 0.539303072917

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: -0.232021063916 Ha
- VMC min minus HF: -3.54782253645 Ha
- VMC tail mean minus HF: -0.115235211017 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
