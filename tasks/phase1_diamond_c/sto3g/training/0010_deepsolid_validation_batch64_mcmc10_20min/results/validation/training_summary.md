# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_batch64_mcmc10_20min`
Created: `2026-05-21T13:36:27.135499+00:00`

## Training Stats

- Rows: 800
- First step: 0
- Last step: 799
- First energy: -41.753407215 Ha
- Last energy: -61.5558509267 Ha
- Minimum energy: -87.8141915505 Ha
- Energy delta: -19.8024437117 Ha
- Energy trend: down
- First variance: 282.084408311
- Last variance: 245.173484047
- Variance trend: down
- Mean pmove: 0.52316015625
- Pmove range: [0.4453125, 0.6140625]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 400
- Tail start step: 400
- Tail energy mean: -59.0591455355 Ha
- Tail energy stderr: 0.168661317578 Ha
- Block count: 5
- Block energy stderr: 1.13675079221 Ha
- Tail variance mean: 380.1651988
- Tail pmove mean: 0.52109765625

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 12.4483458049 Ha
- VMC min minus HF: -13.8099948189 Ha
- VMC tail mean minus HF: 14.9450511961 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
