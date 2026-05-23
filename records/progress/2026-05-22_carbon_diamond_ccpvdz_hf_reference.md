# Carbon Diamond cc-pVDZ HF Reference

Date: 2026-05-22

## Goal

Move the carbon-diamond primitive-cell benchmark from the lightweight `sto-3g`
Hamiltonian to the paper-comparable `ccpvdz` basis before scaling the VMC model,
batch size, or optimizer.

## Configuration

- Experiment:
  `configs/experiment/diamond_c_deepsolid_ccpvdz_hf_reference.yaml`
- DeepSolid template:
  `DeepSolid/config/diamond.py`
- Template input:
  `C,C,3.57,1,ccpvdz`
- Cell:
  1x1x1 primitive carbon diamond at Gamma
- Electrons:
  all-electron, `nelectron=12`, `nelec=(6, 6)`
- Pseudopotential:
  none

## Command

```bash
conda run -n solidnes-deepsolid-jax0430-probe \
  env PYTHONPATH=external/deepsolid:src \
  python scripts/validation/compute_pyscf_pbc_reference.py \
  configs/experiment/diamond_c_deepsolid_ccpvdz_hf_reference.yaml
```

## Result

```text
reference: PySCF PBC KHF via DeepSolid.hf.SCF
converged: True
e_tot_hartree: -74.9757591792
e_tot_per_electron_hartree: -6.2479799316
```

Generated reference file:

```text
tasks/phase1_diamond_c/ccpvdz/references/0018_deepsolid_ccpvdz_hf_reference/results/validation/pyscf_pbc_hf_reference.json
```

Comparison to the DeepSolid paper small C-diamond HF reference:

```text
paper/reference HF: -74.9784 Ha
SolidNES cc-pVDZ HF: -74.9757591792 Ha
difference: +0.0026408208 Ha
```

## Interpretation

The basis upgrade passes the intended sanity check: the same-cell `ccpvdz` HF
energy is within about `2.64 mHa` of the paper benchmark. The small residual
difference is consistent with the slightly different lattice constant
(`3.57 Angstrom` here versus about `3.5738 Angstrom` in the paper) and ordinary
implementation/precision details.

This means the earlier near-HF `sto-3g` result should no longer be compared
directly to the paper energy. Future paper-alignment runs should use this
`ccpvdz` HF reference as the baseline, then scale the network, batch size, and
training length.
