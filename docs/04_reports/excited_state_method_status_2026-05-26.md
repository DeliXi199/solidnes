# Excited-State Method Status, 2026-05-26

## Decision

Use the native FermiNet simultaneous two-state `vmc_overlap` route as the next
excited-state production path.  The current validated production-speed control
is task `0093`:

- `spin_penalty=0.0`
- `observables_s2=false`
- `check_nan=false`
- `reset_if_nan=false`
- batch size `4096`
- 4 A100 40GB GPUs
- 10000 steps in `00:20:36`

This route ran to completion without non-finite CSV rows or diagnostic frames.

## Method 1: Native Simultaneous `vmc_overlap`

This is the route implemented inside FermiNet's native excited-state machinery:
`cfg.system.states=2` and `cfg.optim.objective=vmc_overlap`.

Current status:

- Keep using this method for later excited-state work.
- It is the fastest validated path in this project so far.
- Task `0093` reproduced the expected runtime and completed all 10000 steps.

Known issues:

- Gap estimates are still noisy.  In task `0093`, the final gap was
  `11.831 eV`, while the tail-100 gap mean/median/std was
  `9.788 / 9.132 / 4.925 eV`.
- Raw nonsymmetric overlap diagnostics can have large estimator outliers.  In
  task `0093`, the final raw overlap matrix had a huge off-diagonal value, but
  the symmetric overlap and overlap-penalty diagnostics were stable with final
  off-diagonal `0`.
- The route can suffer rare KFAC non-finite updates.  Task `0091` became
  non-finite after step `3814`; both state energies became NaN from step `3815`.
- Per-step NaN/debug guards are too expensive.  Task `0092` enabled full-tree
  checks and became about `3--4x` slower, so it was cancelled.
- Spin control is not solved.  The beta sweeps and pressure tests showed that
  current spin penalties can reduce some diagnostics but make physical gaps
  noisy or unstable.  Spin penalty work should remain separate from the current
  no-spin production route.

Operational guidance:

- Do not enable `check_nan` or `reset_if_nan` for production-speed native runs.
- Monitor NaNs out of process from logs, `train_stats.csv`, and diagnostic
  arrays.
- Prefer symmetric overlap and overlap-penalty diagnostics over raw
  nonsymmetric overlap.
- Judge gap from tail-window statistics rather than a single final sample.

## Method 2: Fixed-Ground Sequential Penalty

This route first trains or loads a fixed ground-state FermiNet checkpoint, then
optimizes a separate excited state with a fixed-ground overlap penalty.

Current status:

- Do not use this route as the main excited-state path yet.
- Keep it as an experimental branch for later debugging.

Known issues:

- The asymmetric fixed-ground route in task `0089` completed, but the physical
  excited-state energy fell below the fixed ground reference.  The final
  `physical_energy` was `-75.420519 Ha`, while the fixed ground reference was
  `-75.411863 Ha`; this gives an unphysical negative excitation gap.
- Task `0089` also retained a nontrivial fixed-ground overlap around `0.112`,
  so the state was not cleanly separated from the fixed reference.
- The symmetric fixed-ground route in task `0090` required extra fixes for
  JAX donated-buffer aliasing and was cancelled before completion at step
  `18635`.
- In task `0090`, the reported fixed-ground overlap often collapsed to exact
  zero because the product of excited-sample and ground-sample overlap
  estimates changed sign or crossed zero.  This makes the penalty signal
  unreliable as a production objective.
- Both fixed-ground variants are slower than the native route.  Task `0089`
  took `02:28:34` for 20000 steps, and task `0090` consumed `02:44:02` before
  cancellation at 18636 written rows.

Operational guidance:

- Do not base the next production excited-state calculations on fixed-ground
  overlap yet.
- Before revisiting it, define a better signed/complex-safe overlap estimator,
  validate it on a small system, and add analysis that directly compares
  physical excited energy to the fixed ground reference.
- Keep fixed-ground work independent from the native simultaneous route so
  failures there do not block native-route production runs.

## Current Baseline Numbers

Native no-spin route, task `0093`:

- Completed `10000` rows, no non-finite rows or frames.
- Final scalar energy: `-74.981740 Ha`.
- Tail-100 scalar energy mean: `-75.029858 Ha`.
- Final state energies: `[-75.126656, -74.691872] Ha`.
- Final gap: `11.831 eV`.
- Tail-100 gap mean/median/std: `9.788 / 9.132 / 4.925 eV`.

Fixed-ground asymmetric route, task `0089`:

- Completed `20000` rows.
- Final physical energy: `-75.420519 Ha`.
- Fixed ground reference: `-75.411863 Ha`.
- Final fixed-ground overlap: `0.112159`.

Fixed-ground symmetric route, task `0090`:

- Cancelled after `18636` written rows.
- Final physical energy: `-75.255753 Ha`.
- Fixed ground reference: `-75.411863 Ha`.
- Final fixed-ground overlap: `0.0`, but component estimates remained
  nonzero/sign-changing.
