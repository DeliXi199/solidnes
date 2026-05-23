# FermiNet PBC-HF Pretraining Patch

Status: local SolidNES extension to upstream FermiNet.

Patch file:

```text
patches/ferminet/pbc_hf_pretraining.patch
```

Upstream FermiNet commit:

```text
c4312c315dda1c5728994ba89629744f71c6eb66
```

## Purpose

Upstream FermiNet supports molecular Hartree-Fock orbital pretraining, but the
periodic-boundary-condition solid route does not provide a PySCF PBC-HF
pretraining target. This patch adds a PBC-HF orbital pretraining path for the
SolidNES FermiNet backend.

## Upstream FermiNet Files Modified

- `ferminet/base_config.py`
  - Adds local PBC-HF pretraining config fields:
    `pretrain.pbc`, `pretrain.twist`, `pretrain.cell_precision`,
    `pretrain.cell_exp_to_discard`, `pretrain.restricted`,
    `pretrain.learning_rate`, and `pretrain.mcmc_steps`.
  - Documents `pretrain.method: pbc_hf` as the local SolidNES PBC route.

- `ferminet/pretrain.py`
  - Adds `PbcScf` and `get_pbc_hf`.
  - Builds a PySCF PBC cell from FermiNet PBC molecule/lattice data.
  - Runs PySCF PBC KHF/KUHF with the configured twist.
  - Evaluates occupied PBC-HF orbital matrices at walker coordinates.
  - Adds `make_pbc_pretrain_step` and `pretrain_hartree_fock_pbc`.

- `ferminet/train.py`
  - Dispatches `pretrain.method: pbc_hf` or `pretrain.pbc: true` to the PBC-HF
    path.
  - Keeps upstream molecular `pretrain.method: hf` behavior unchanged.

Related existing local patch:

- `ferminet/networks.py`
  - Already patched separately by `patches/ferminet/folx_tile_broadcast.patch`
    to replace a `tile` call with `broadcast_to` in
    `construct_symmetric_features`.

## SolidNES Files Added Or Updated

- `src/solidnes/backends/ferminet_adapter.py`
  - Propagates PBC-HF pretraining options from SolidNES YAML into the FermiNet
    config.
  - Uses system basis/twist/lattice and optional pseudopotential metadata.
  - Forces PBC pretraining `scf_fraction` to `0.0` unless explicitly overridden,
    because PySCF PBC orbital evaluation is host-side and not used inside JIT
    MCMC.

- `configs/train/ground_state_ferminet_pbc_kfac_folx_batch4096_iter20000_pbc_hf_pretrain.yaml`
  - New training config enabling `pretrain_method: pbc_hf`.

- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml`
  - New ready-to-submit paper-geometry diamond pilot using PBC-HF pretraining.

- `records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md`
  - Implementation and verification record.

## Design Note

This follows DeepSolid's practical design rather than upstream FermiNet's
molecular-HF implementation. PySCF PBC orbital evaluation is host-side, so the
HF target orbital matrices are evaluated once per pretraining iteration and
then passed as fixed targets into a pmapped JAX optimization step. The walker
refresh uses the neural network wavefunction.

## Verification

```bash
python -m py_compile \
  external/ferminet/ferminet/base_config.py \
  external/ferminet/ferminet/pretrain.py \
  external/ferminet/ferminet/train.py \
  src/solidnes/backends/ferminet_adapter.py

JAX_PLATFORMS=cpu python scripts/backends/build_ferminet_config.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml

JAX_PLATFORMS=cpu python scripts/backends/run_ferminet_train.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot.yaml \
  --build-only
```

Additional local probes passed:

- Paper-geometry `ccpvdz` PBC-HF target converged at
  `E_HF = -74.977894072367 Ha`.
- Minimal CPU one-step PBC pretraining probe passed with a `sto-3g` target:
  `pbc_pretrain_one_step=ok`, walker shape `(1, 2, 36)`,
  `E_HF = -74.0041967307 Ha`.
