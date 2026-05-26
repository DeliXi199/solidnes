# Reference Notes: DeepQMC Penalty Excited States

Last updated: 2026-05-24

## Source Snapshot

Reference repository:

```text
https://github.com/deepqmc/deepqmc
```

Local ignored checkout:

```text
external/deepqmc/
```

Inspected revision:

```text
f9e1ff5 add PseudoHamiltonians (#237)
```

This checkout is a reference source only. It is under ignored `external/` and
does not consume a SolidNES run number. No `tasks/` bundle is needed until a
build, smoke, experiment, evaluation, analysis, SLURM plan, log, result, or
validation artifact is produced.

## Why This Reference Matters

DeepQMC is the public implementation path for the Szabo and Noe JCTC 2024
penalty-based excited-state VMC method. Its README states that excited states
are obtained with a penalty-based optimization approach and that a spin penalty
can target a fixed spin sector.

DeepQMC is molecular, not periodic-solid-first. For SolidNES, it should be used
as an algorithm reference rather than as the primary backend. The current
SolidNES ground-state and pretraining route is already built around FermiNet
PBC on carbon diamond.

## Key DeepQMC Entry Points

```text
external/deepqmc/README.md
external/deepqmc/doc/cli.rst
external/deepqmc/src/deepqmc/conf/task/train_excited_psiformer.yaml
external/deepqmc/src/deepqmc/loss/overlap.py
external/deepqmc/src/deepqmc/loss/loss_function.py
external/deepqmc/src/deepqmc/loss/clip.py
external/deepqmc/src/deepqmc/loss/spin.py
external/deepqmc/src/deepqmc/sampling/combined_samplers.py
external/deepqmc/src/deepqmc/wf/base.py
external/deepqmc/src/deepqmc/train.py
```

## Algorithm Pieces To Borrow

### Multi-state parameterization

DeepQMC initializes one parameter tree per electronic state:

```text
params shape: [electronic_states, ...]
```

The code optionally merges selected parameter subtrees between states. For
SolidNES, the first version should use separate state parameter trees because
that is easiest to reason about and diagnose. Parameter sharing can wait until
the two-state periodic objective is stable.

### Multi-state sampling

DeepQMC wraps an electron sampler with `MultiElectronicStateSampler`, which
samples each state in parallel with a leading state axis. Conceptually:

```text
samples[j] ~ |psi_j|^2
```

SolidNES should mirror this shape at the diagnostics/objective layer:

```text
positions shape: [states, walkers, nelec, ndim]
```

For FermiNet PBC integration, the first practical route is likely to maintain
one walker population per state and evaluate all state wavefunctions on all
state samples when estimating overlaps.

### Overlap estimator

DeepQMC evaluates every wavefunction on every state's samples:

```text
psi[i, j, k] = psi_i(r_k sampled from |psi_j|^2)
ratio[i, j, k] = psi_i(r_jk) / psi_j(r_jk)
```

The non-symmetric Monte Carlo overlap estimate is:

```text
x_ij = mean_{r ~ |psi_j|^2} ratio[i, j, r]
```

DeepQMC then symmetrizes with a clipped geometric mean:

```text
S_ij = sign(x_ij) * sqrt(clip_min(x_ij * x_ji, 0))
```

The DeepQMC docstring describes a `[0, 1]` clamp; the inspected implementation
uses a lower-bound clip. SolidNES should keep this distinction explicit in
tests before choosing the exact numerical guard.

The overlap loss is the sum of squared upper-triangle overlaps:

```text
L_overlap = sum_{i < j} S_ij^2
```

This is the main part to port first into `src/solidnes/excited_states/`.

### Penalty objective

DeepQMC's loss combines state energies and overlap penalty:

```text
L = mean_state_energy + alpha * L_overlap
```

The excited-state config uses:

```text
alpha: 4.0
scale_overlap_by: max_gap_std
min_gap_scale_factor: 0.001
```

SolidNES should not hard-code these as final production defaults. The first
implementation should expose them in config and log them with every run.

### Gradient scaling

The Szabo-Noe improvement implemented in DeepQMC is not just the scalar penalty
term; the overlap-gradient tangent can be scaled by:

```text
none
energy_gap
energy_std
max_gap_std
```

The most relevant first choice is `max_gap_std`, because it avoids an
over-aggressive overlap gradient when states are close or noisy. SolidNES
should implement this at the objective-utility level before touching a long
training run.

### Clipping and masks

DeepQMC clips both local energies and wavefunction ratios:

```text
median energy clipping
psi-ratio clipping for overlap gradients
```

SolidNES should port the ratio clipping idea before any serious optimization,
because overlap ratios can have heavy tails and can destabilize gradients.

### Spin penalty

DeepQMC has an optional spin penalty:

```text
L += spin_penalty * <S^2>
```

For SolidNES periodic diamond Gamma, the first controlled two-state probe
initially deferred this while the overlap method was being stabilized. The
native FermiNet path now exposes this as `spin_penalty`, mapped to
`cfg.optim.spin_energy`, with `observables_s2` writing `s2_matrix.npy`.
Spin-targeted runs still need an explicit target sector and penalty strength in
the experiment YAML.

## Current SolidNES/FermiNet Mapping

SolidNES should not switch to DeepQMC as the backend. The current route should
remain:

```text
FermiNet PBC ground-state route
-> SolidNES-owned two-state penalty objective
-> FermiNet PBC local-energy integration
-> carbon diamond Gamma two-state probe
```

Existing FermiNet code already has a molecular overlap objective:

```text
external/ferminet/ferminet/loss.py
```

But its PBC Hamiltonian currently rejects excited states:

```text
external/ferminet/ferminet/pbc/hamiltonian.py
```

This means the SolidNES integration must explicitly bridge PBC local-energy
evaluation and state-wise objective logic. It is not enough to set
`cfg.system.states = 2`.

## Proposed SolidNES Implementation Order

1. Add a small `src/solidnes/excited_states/` package.
2. Implement backend-independent overlap utilities:
   - all-state wavefunction ratio interface;
   - clipped geometric-mean overlap symmetrization;
   - upper-triangle squared-overlap loss;
   - overlap diagnostics.
3. Implement penalty-scale utilities:
   - no scaling;
   - energy-gap scaling;
   - energy-std scaling;
   - max-gap-std scaling.
4. Add lightweight deterministic tests or script-level checks using synthetic
   ratio matrices, without FermiNet or SLURM.
5. Add FermiNet PBC adapter scaffolding only after the objective utilities are
   tested in isolation.
6. Create the first numbered task bundle only for the first build/smoke/run
   step that produces project-owned artifacts.

## First Probe Shape

The first physical probe should stay narrow:

```text
System: carbon diamond primitive cell
K point: Gamma
Basis: same cc-pVDZ workflow as the validated ground-state route
States: 2
Backend: FermiNet PBC
Pretraining: optional, initially documented as a branch rather than default
```

Minimum diagnostics:

```text
state_energy[0]
state_energy[1]
excitation_gap_hartree
excitation_gap_ev
overlap_matrix
max_abs_offdiag_overlap
overlap_penalty
penalty_alpha
overlap_gradient_scale
state_ordering
collapse_flag
```

## Deferred Items

- DeepQMC as a runnable SolidNES backend.
- Spin penalty in the first periodic probe.
- CAS excited-state pretraining parity. SolidNES has PBC-HF pretraining, but a
  periodic excited-state pretraining target needs separate design.
- Parameter sharing between state networks.
- Non-Gamma twists, finite-size scaling, and material tests before the diamond
  two-state diagnostic path is stable.

## Immediate Next Action

Implement the backend-independent overlap/penalty utility scaffold in
`src/solidnes/excited_states/` and verify it with no-compute synthetic checks.
Do not create a numbered task bundle until a build/smoke/check artifact needs
to be recorded under `tasks/`.
