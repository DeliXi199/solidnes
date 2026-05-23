# 2026-05-23 FermiNet PBC-HF Pretraining

Implemented true PySCF PBC-HF orbital pretraining for the local FermiNet PBC
route.

## Implementation

- Added FermiNet pretrain config defaults for PBC-HF:
  `pretrain.pbc`, `pretrain.twist`, `pretrain.cell_precision`,
  `pretrain.cell_exp_to_discard`, `pretrain.restricted`,
  `pretrain.learning_rate`, and `pretrain.mcmc_steps`.
- Added `PbcScf` / `get_pbc_hf` in `external/ferminet/ferminet/pretrain.py`.
  This builds a PySCF PBC cell from the FermiNet PBC molecule and lattice,
  runs KHF/KUHF, selects occupied orbitals per spin, wraps electron coordinates
  into the primitive cell, and evaluates occupied HF orbital matrices.
- Added `pretrain_hartree_fock_pbc`, following DeepSolid's robust route:
  evaluate PySCF PBC target matrices on the host each iteration and pass those
  fixed targets into a pmapped network-orbital MSE update. The walker refresh
  uses the neural wavefunction rather than putting PySCF PBC orbital evaluation
  inside JIT.
- Updated `external/ferminet/ferminet/train.py` so `pretrain.method:
  pbc_hf` or `pretrain.pbc: true` dispatches to the PBC-HF path.
- Updated the SolidNES FermiNet adapter to propagate system basis, twist,
  lattice, and PBC pretraining options from YAML.

## New Configs

- `configs/train/ground_state_ferminet_pbc_kfac_folx_batch4096_iter20000_pbc_hf_pretrain.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml`

## Verification

Commands run in `.venv/ferminet-jax0101-cuda12`:

```bash
python -m py_compile \
  external/ferminet/ferminet/pretrain.py \
  external/ferminet/ferminet/train.py \
  external/ferminet/ferminet/base_config.py \
  src/solidnes/backends/ferminet_adapter.py

python scripts/backends/build_ferminet_config.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml

python scripts/backends/run_ferminet_train.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml \
  --build-only
```

Additional local CPU probes:

- Paper-geometry `ccpvdz` PySCF PBC-HF target converged:
  `E_HF = -74.977894072367 Ha`; target matrix shapes were `(2, 6, 6)` for
  both spin channels, with zero imaginary component at Gamma.
- One minimal PBC pretraining update passed on CPU with a `sto-3g` target:
  `pbc_pretrain_one_step=ok`, walker shape `(1, 2, 36)`, and
  `E_HF = -74.0041967307 Ha`.

## Notes

This is intentionally closer to DeepSolid's stable implementation than to
upstream FermiNet's molecular HF path. The molecular path can evaluate GTOs in
JAX and therefore supports HF-mixture MCMC inside JIT. PySCF PBC orbital
evaluation is host-side, so the PBC path samples walkers from the neural
wavefunction during pretraining.
