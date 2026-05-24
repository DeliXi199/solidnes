# JAX PBC GTO Target Implementation Notes

Created: `2026-05-24`

## Code Paths

- `src/solidnes/backends/ferminet_pbc_gto.py`
  - Minimal Gamma-compatible periodic GTO evaluator using finite lattice image
    sums.
- `src/solidnes/backends/ferminet_pbc_hf_target.py`
  - Device-side PBC HF target wrapper used by the local FermiNet pretraining
    patch.
- `external/ferminet/ferminet/pretrain.py`
  - Adds `target_backend` dispatch and avoids `np.asarray(data.positions)` for
    device-side target evaluation.
- `external/ferminet/ferminet/train.py`
  - Writes `pretrain_stats.csv` and passes target backend options.
- `src/solidnes/backends/ferminet_adapter.py`
  - Maps YAML fields into the FermiNet config and keeps `jax_pbc_gto` gated to
    `sto-3g`.

## Validation

- AO/MO PySCF comparison passed for diamond Gamma `sto-3g`, `image_cutoff=2`:
  AO max abs `3.28e-7`, occupied MO max abs `3.03e-7`.
- CPU pretrain regression passed before GPU submission.
- GPU benchmark `0048` completed on scheduled A100 80GB node.
- Fair PySCF `sto-3g` GPU comparison `0049` completed on the same node class.

## Existing Local Dirty File

`external/ferminet/ferminet/networks.py` contains a pre-existing local change:
`jnp.tile` was replaced with `jnp.broadcast_to` in symmetric feature
construction. This was observed and preserved; it is not part of the JAX PBC GTO
target implementation.
