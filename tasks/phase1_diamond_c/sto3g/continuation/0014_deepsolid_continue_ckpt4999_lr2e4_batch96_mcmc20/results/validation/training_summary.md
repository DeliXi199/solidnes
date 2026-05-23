# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20`
Created: `2026-05-21T16:01:21.602420+00:00`

## Training Stats

- Rows: 5000
- First step: 5000
- Last step: 9999
- First energy: -72.6434480692 Ha
- Last energy: -73.9764365996 Ha
- Minimum energy: -77.6006017258 Ha
- Energy delta: -1.33298853042 Ha
- Energy trend: down
- First variance: 105.498448229
- Last variance: 78.585786973
- Variance trend: down
- Mean pmove: 0.527377916667
- Pmove range: [0.4640625, 0.582291666667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 2500
- Tail start step: 7500
- Tail energy mean: -73.095331312 Ha
- Tail energy stderr: 0.0149495922261 Ha
- Block count: 10
- Block energy stderr: 0.0570580107243 Ha
- Tail variance mean: 51.5340198803
- Tail pmove mean: 0.529090208333

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 0.027760131962 Ha
- VMC min minus HF: -3.59640499416 Ha
- VMC tail mean minus HF: 0.908865419599 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
