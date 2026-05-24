# cc-pVDZ JAX PBC GTO Correctness and Backend Comparison

Created: `2026-05-24`

## Correctness

Diamond primitive Gamma, `ccpvdz`, 32 random in-cell points:

| Image cutoff | AO max abs | AO RMS abs | Worst AO column |
| ---: | ---: | ---: | --- |
| 1 | 4.18252696413e-02 | 4.83290193214e-03 | `0 C 3px` |
| 2 | 6.98569225853e-05 | 5.97461929903e-06 | `0 C 3px` |
| 3 | 1.12106182670e-09 | 7.99735711140e-11 | `0 C 3px` |

At `image_cutoff=3`, occupied MO comparison against PySCF HF coefficients:

- MO max abs: `8.51067952223e-10`
- MO RMS abs: `1.00005713936e-10`
- HF converged: `True`
- PySCF HF energy: `-74.9778942359 Ha`

The d-shell AO groups are not the source of error. At cutoff 3, the d-shell
group errors are at floating-point noise scale (`~1e-15`). The dominant errors
at lower cutoffs come from diffuse `3p`/`3s` image truncation.

## GPU Benchmark

`0050`: `jax_pbc_gto`, `ccpvdz`, `image_cutoff=3`, `batch_size=512`,
`pretrain_iterations=100`, scheduled A100 80GB.

- Job: `128288`
- Status: `COMPLETED 0:0`
- Loss: `2.14452184636 -> 0.01784817926`
- Mean pmove: `0.841172`
- Mean steady step seconds: `0.023191`
- Median steady step seconds: `0.023308`
- P90 steady step seconds: `0.024137`
- Mean target eval seconds: `0.015405`
- Mean JAX update seconds: `0.007429`
- Target transfer seconds: `0.000000`

## Comparison to PySCF cc-pVDZ Target

`0047`: `pyscf_pbc`, `ccpvdz`, `batch_size=512`, `pretrain_iterations=100`,
scheduled A100 80GB. The 0047 run predates per-step CSV logging, so its steady
statistics are computed from the logged pretrain rows after excluding step 0.

| Metric | 0047 PySCF PBC | 0050 JAX PBC GTO | Speedup |
| --- | ---: | ---: | ---: |
| Mean step seconds | 0.058384 | 0.023191 | 2.52x |
| Median step seconds | 0.058559 | 0.023308 | 2.51x |
| P90 step seconds | 0.058849 | 0.024137 | 2.44x |
| Mean target eval seconds | 0.051142 | 0.015405 | 3.32x |
| Mean JAX update seconds | 0.006495 | 0.007429 | 0.87x |

Loss and pmove remain comparable:

- 0047 PySCF target loss: `2.14205 -> 0.0179985`
- 0050 JAX target loss: `2.14452184636 -> 0.01784817926`
- 0047 mean pmove: `0.844638`
- 0050 mean pmove: `0.841172`

## Decision

The `jax_pbc_gto` target is ready to use for diamond Gamma cc-pVDZ pretraining
benchmarks with:

```yaml
pretrain_target_backend: jax_pbc_gto
pretrain_basis: ccpvdz
pretrain_jax_pbc_image_cutoff: 3
```

Keep the adapter guard that requires `image_cutoff >= 3` for cc-pVDZ. The next
production step should be a longer pretrain-plus-training pilot, not more small
100-step target timing benchmarks.
