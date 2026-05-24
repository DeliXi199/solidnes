# 2026-05-24 FermiNet PBC-HF Pretraining Milestone

## Conclusion

The FermiNet PBC-HF pretraining milestone is complete at the project level for
the current carbon-diamond Gamma, cc-pVDZ workflow.

This means:

- The PBC-HF pretraining code path is implemented and integrated into the
  SolidNES FermiNet route.
- GPU runs confirm that pretraining executes, records diagnostics, and hands
  control to downstream KFAC/FOLX VMC training.
- The JAX PBC GTO target backend is validated for the current diamond Gamma
  `ccpvdz` setup and is the preferred pretraining target backend for this
  workflow.

This does not yet mean:

- PBC-HF pretraining is proven to improve total wall-clock efficiency in every
  matched run.
- The `jax_pbc_gto` backend is validated for non-Gamma twists, different
  cells, pseudopotentials, or larger basis sets.

## Evidence

Implementation record:

```text
records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
```

Key implementation facts:

- Added true PySCF PBC-HF orbital pretraining for the local FermiNet PBC route.
- Added SolidNES adapter support for system basis, twist, lattice, and PBC
  pretraining options.
- Added host-side PySCF PBC target evaluation and a JAX PBC GTO target backend.
- Build-only checks and a local CPU one-step PBC pretraining probe passed.

GPU target/backend validation:

```text
0047: PySCF PBC-HF cc-pVDZ pretraining benchmark
0048: JAX PBC GTO sto-3g benchmark
0049: PySCF PBC-HF sto-3g benchmark
0050: JAX PBC GTO cc-pVDZ benchmark and correctness check
```

Important numbers:

- `0047` PySCF cc-pVDZ pretraining loss dropped from `2.14205` to
  `0.0179985`; steady mean step was `0.058384 s`.
- `0048` JAX PBC GTO sto-3g was `5.39x` faster per total pretrain step and
  `19.19x` faster in target evaluation than the PySCF sto-3g target.
- `0050` JAX PBC GTO cc-pVDZ validation at image cutoff `3` passed with AO max
  abs `1.12e-9` and occupied-MO max abs `8.51e-10`.
- `0050` cc-pVDZ JAX PBC GTO measured mean step `0.023191 s` versus
  `0.058384 s` for the PySCF cc-pVDZ benchmark, about `2.52x` faster overall.

Training integration validation:

```text
0053, 0054: 1000-step pretrain + KFAC/FOLX runs
0055, 0056: matched 1000-step no-pretrain controls
0059--0062: 20-minute pretrain-length sweep
0057, 0058: 20-minute no-pretrain controls
```

Key training facts:

- `0053` and `0054` completed full-node 2-GPU runs with 1000 pretrain steps and
  1000 KFAC/FOLX steps without tracebacks.
- Their pretrain losses fell from about `2.141` to about `0.0061--0.0062`.
- Fixed-iteration 1000-step KFAC/FOLX comparison favored pretraining:
  - pretrained last-50 means: `-75.254433 Ha`, `-75.250662 Ha`
  - no-pretrain last-50 means: `-75.167610 Ha`, `-75.172448 Ha`
- Matched wall-clock timebox results were mixed:
  - no-pretrain controls `0057`/`0058` tail-100 means:
    `-75.305930 Ha`, `-75.310207 Ha`
  - pretrain sweep `0059--0062` tail-100 means:
    `-75.286429 Ha`, `-75.311533 Ha`, `-75.274833 Ha`, `-75.279337 Ha`
  - among the tested pretrain lengths, 250 steps was the best timeboxed point,
    but it only matched the no-pretrain wall-clock controls within the current
    noisy short-run evidence.

## Project Decision

Treat the PBC-HF pretraining implementation and diamond-Gamma validation as
complete.

Use this conclusion going forward:

```text
PBC-HF pretraining is a working, instrumented FermiNet option for the current
diamond Gamma cc-pVDZ workflow. It improves fixed-iteration early training in
the 1000-step comparison, but it is not yet a universal production default on
wall-clock efficiency. Keep it as a validated option while moving to the first
controlled NES-VMC excited-state probe.
```

## Next Step

Move the main project focus from pretraining implementation to the first
controlled periodic excited-state/NES-VMC extension. For any future production
ground-state run, choose between:

- no pretraining when the priority is raw wall-clock progress, or
- short JAX PBC GTO pretraining, especially around the tested 250-step region,
  when initialization quality is the priority.
