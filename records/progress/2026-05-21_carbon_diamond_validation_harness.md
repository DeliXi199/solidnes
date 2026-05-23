# Progress: Carbon Diamond Validation Harness

Date: 2026-05-21

## Summary

SolidNES now has a repeatable validation harness for the current carbon-diamond
DeepSolid setup.

The harness computes:

- Same-cell PySCF PBC KHF reference.
- DeepSolid training statistics summary.
- Energy/variance/pmove trends.
- VMC-minus-HF comparison for the same small Hamiltonian.

This validates the workflow and diagnostics. It does not prove physical
accuracy for carbon diamond yet.

## Added Files

Configs:

```text
configs/sampler/metropolis_deepsolid_validation.yaml
configs/train/ground_state_deepsolid_validation_short.yaml
configs/train/ground_state_deepsolid_validation_stable.yaml
configs/experiment/diamond_c_deepsolid_validation_short.yaml
configs/experiment/diamond_c_deepsolid_validation_stable.yaml
```

Scripts:

```text
scripts/validation/compute_pyscf_pbc_reference.py
scripts/validation/summarize_deepsolid_validation.py
```

Adapter update:

```text
src/solidnes/backends/deepsolid_adapter.py
```

The adapter now maps optional training config fields for Adam learning rate,
learning-rate decay/delay, local-energy clipping, ministeps, and checkpoint
frequency.

## HF Reference

Command:

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/validation/compute_pyscf_pbc_reference.py \
  configs/experiment/diamond_c_deepsolid_validation_stable.yaml
```

Output:

```text
reference: PySCF PBC KHF via DeepSolid.hf.SCF
converged: True
e_tot_hartree: -74.0041967316
e_tot_per_electron_hartree: -6.1670163943
```

Reference file:

```text
tasks/phase1_diamond_c/sto3g/training/0007_deepsolid_validation_stable/results/validation/pyscf_pbc_hf_reference.json
```

## 20-Step Validation

SLURM job:

```text
120704
state COMPLETED
exit code 0:0
elapsed 00:00:35
node gpu40904
```

Summary:

```text
rows: 20
energy_first: -48.41835816 Ha
energy_last: -27.2161501781 Ha
energy_delta: +21.2022079819 Ha
variance_delta: -3039.32332
pmove_mean: 0.54625
last_minus_hf: +46.7880465535 Ha
```

Interpretation:

The run is finite and MCMC acceptance is useful, but the energy trend is not an
accuracy pass.

## 50-Step Stable Validation

Changes from the first validation:

```text
batch_size: 16
iterations: 50
learning_rate: 0.005
```

SLURM job:

```text
120710
state COMPLETED
exit code 0:0
elapsed 00:00:36
node gpu40904
```

Summary:

```text
rows: 50
energy_first: -28.2425761797 Ha
energy_last: -41.5337426643 Ha
energy_delta: -13.2911664846 Ha
energy_min: -49.0733496833 Ha
variance_delta: -15.711579288
pmove_mean: 0.5425
pmove_range: [0.3875, 0.7125]
last_minus_hf: +32.4704540672 Ha
min_minus_hf: +24.9308470482 Ha
```

Reports:

```text
tasks/phase1_diamond_c/sto3g/training/0006_deepsolid_validation_short/results/validation/training_summary.md
tasks/phase1_diamond_c/sto3g/training/0007_deepsolid_validation_stable/results/validation/training_summary.md
```

## Conclusion

Accuracy can now be checked mechanically for this setup, but the current short
validation runs do not yet demonstrate accurate or converged carbon-diamond
ground-state energy.

The good news is that the second run shows the expected direction:

- finite training statistics,
- useful MCMC acceptance,
- lower final energy,
- lower final variance.

The gap to HF is still much too large for an accuracy claim.

## Next

Do not move to a physical accuracy claim yet. The next validation step should
increase physics/training quality before NES-VMC:

```text
1. Add HF/pretraining support or restore-compatible initialization.
2. Increase model/batch/iterations enough to approach the HF baseline.
3. Add block-averaged uncertainty estimates.
4. Try the less toy `ccpvdz` carbon diamond config after the harness is stable.
```
