# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt49999_lr2e5_batch384_mcmc20_iter150000`
Created: `2026-05-22T14:35:03.646805+00:00`

## Training Stats

- Rows: 100000
- First step: 50000
- Last step: 149999
- First energy: -75.2215320914 Ha
- Last energy: -75.2704621344 Ha
- Minimum energy: -78.867323555 Ha
- Energy delta: -0.0489300430497 Ha
- Energy trend: down
- First variance: 6.31477950905
- Last variance: 10.4466143737
- Variance trend: up
- Mean pmove: 0.539358998698
- Pmove range: [0.4796875, 0.574739583333]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 50000
- Tail start step: 100000
- Tail energy mean: -75.1982415149 Ha
- Tail energy stderr: 0.000637676097298 Ha
- Block count: 5
- Block energy stderr: 0.00332332240829 Ha
- Tail variance mean: 8.05559367299
- Tail pmove mean: 0.539440192708

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: -0.294702955249 Ha
- VMC min minus HF: -3.89156437581 Ha
- VMC tail mean minus HF: -0.222482335701 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
