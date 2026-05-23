# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_ccpvdz_paper_net_continue_ckpt9999_lr1e4_batch384_mcmc20_iter30000`
Created: `2026-05-22T06:21:32.229212+00:00`

## Training Stats

- Rows: 20000
- First step: 10000
- Last step: 29999
- First energy: -74.3952468354 Ha
- Last energy: -74.7065472189 Ha
- Minimum energy: -81.3313879034 Ha
- Energy delta: -0.311300383534 Ha
- Energy trend: down
- First variance: 21.3084702461
- Last variance: 15.2261081901
- Variance trend: down
- Mean pmove: 0.539191595052
- Pmove range: [0.488411458333, 0.590494791667]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 10000
- Tail start step: 20000
- Tail energy mean: -74.9086210088 Ha
- Tail energy stderr: 0.00211067141113 Ha
- Block count: 5
- Block energy stderr: 0.0211787375811 Ha
- Tail variance mean: 16.5876240636
- Tail pmove mean: 0.5392928125

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.9757591792 Ha
- VMC last minus HF: 0.269211960258 Ha
- VMC min minus HF: -6.35562872421 Ha
- VMC tail mean minus HF: 0.0671381703184 Ha

## Scope

This is a validation harness for the configured carbon `ccpvdz` setup.
It checks finite execution, trends, and HF comparison for the same
setup; it is not by itself a production carbon benchmark.
