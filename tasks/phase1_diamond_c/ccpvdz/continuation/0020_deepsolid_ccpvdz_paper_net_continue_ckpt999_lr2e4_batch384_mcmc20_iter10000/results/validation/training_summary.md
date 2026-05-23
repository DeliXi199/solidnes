# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt999_lr2e4_batch384_mcmc20_iter10000`
Created: `2026-05-22T04:08:20.151109+00:00`

## Training Stats

- Rows: 9000
- First step: 1000
- Last step: 9999
- First energy: -72.8129042284 Ha
- Last energy: -74.6358875912 Ha
- Minimum energy: -79.2683728547 Ha
- Energy delta: -1.82298336278 Ha
- Energy trend: down
- First variance: 40.130963543
- Last variance: 23.2866413615
- Variance trend: down
- Mean pmove: 0.538286993634
- Pmove range: [0.481770833333, 0.592838541667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 4500
- Tail start step: 5500
- Tail energy mean: -74.2934236537 Ha
- Tail energy stderr: 0.00508683652463 Ha
- Block count: 5
- Block energy stderr: 0.0585045710779 Ha
- Tail variance mean: 31.3324361843
- Tail pmove mean: 0.538136111111

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: 0.339871587953 Ha
- VMC min minus HF: -4.29261367556 Ha
- VMC tail mean minus HF: 0.682335525433 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
