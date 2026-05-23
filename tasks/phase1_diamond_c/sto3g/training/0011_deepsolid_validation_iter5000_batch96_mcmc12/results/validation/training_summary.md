# Carbon Diamond Validation Summary

Experiment: `diamond_c_deepsolid_validation_iter5000_batch96_mcmc12`
Created: `2026-05-21T13:56:31.868021+00:00`

## Training Stats

- Rows: 3197
- First step: 0
- Last step: 3196
- First energy: -57.7949294657 Ha
- Last energy: -72.1752396199 Ha
- Minimum energy: -75.8069906299 Ha
- Energy delta: -14.3803101541 Ha
- Energy trend: down
- First variance: 921.599857269
- Last variance: 81.5122409513
- Variance trend: down
- Mean pmove: 0.52232887603
- Pmove range: [0.420138888889, 0.603298611111]
- Finite checks: True

## Tail / Block Estimate

- Tail rows: 1599
- Tail start step: 1598
- Tail energy mean: -70.8751459188 Ha
- Tail energy stderr: 0.029997618665 Ha
- Block count: 5
- Block energy stderr: 0.228480776062 Ha
- Tail variance mean: 120.890828394
- Tail pmove mean: 0.522380524286

## HF Reference

- Reference: PySCF PBC KHF via DeepSolid.hf.SCF
- Converged: True
- HF total energy: -74.0041967316 Ha
- VMC last minus HF: 1.82895711173 Ha
- VMC min minus HF: -1.80279389835 Ha
- VMC tail mean minus HF: 3.12905081281 Ha

## Scope

This is a validation harness for the small carbon `sto-3g` setup. It
checks finite execution, trends, and HF comparison for the same setup;
it is not a production carbon benchmark.
