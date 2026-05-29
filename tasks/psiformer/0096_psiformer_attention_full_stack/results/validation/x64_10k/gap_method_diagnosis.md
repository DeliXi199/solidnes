# Gap and Excited-State Method Diagnosis

Date: 2026-05-28

## Current result

The 10k x64 attention runs do not show a consistently small diamond Gamma gap.
The final single-step gaps are low because the final Monte Carlo sample is noisy:

- upstream/FermiNet x64: final gap 4.046 eV; last-1000 mean 8.349 +/- 6.884 eV
- fused-QKV x64: final gap 4.461 eV; last-1000 mean 6.881 +/- 3.120 eV

For the final 8000 steps, 1000-step bin means mostly sit around 6.5-8.5 eV.
The issue is therefore not a stable low gap, but an unstable excited-state energy
estimate with large variance.

## Evidence from diagnostics

- State ordering flips only at the beginning:
  - upstream/FermiNet x64: 14 / 10000 frames differ from [0, 1], last flip at step 14
  - fused-QKV x64: 9 / 10000 frames differ from [0, 1], last flip at step 9
- Final symmetric overlap is small:
  - upstream/FermiNet x64: off-diagonal overlap 0.0553, penalty 0.00305
  - fused-QKV x64: off-diagonal overlap 0.0537, penalty 0.00289
- This does not look like late root-flipping or simple overlap collapse.
- The main visible problem is high excited-state local-energy variance and weak
  root targeting.

## Method comparison

SolidNES/FermiNet native and DeepQMC both use the Szabo/Entwistle-style
penalty method:

- energy objective plus alpha * overlap penalty
- all-state wavefunction-ratio estimates from samples of each state
- symmetric overlap estimate from paired Monte Carlo expectations
- psi-ratio clipping in the overlap tangent
- optional scaling of the overlap gradient by max(energy gap, energy std)
- optional energy-based state ordering

Important differences from DeepQMC's recommended excited-state workflow:

1. DeepQMC explicitly recommends CASSCF/CAS pretraining for multi-state
   excited-state runs. The current SolidNES production run uses
   pretrain_iterations: 0 and pretrain_method: null.
2. DeepQMC's excited PsiFormer config is designed for much longer optimization
   (default 100000 steps). The current comparison has only completed 10000 steps,
   with continuation to 15000 still running.
3. DeepQMC is molecular-first. The current SolidNES case is periodic all-electron
   primitive diamond at Gamma, so the computed two-state VMC excitation is not
   automatically equivalent to an experimental quasiparticle band gap without
   finite-size, twist, and root-targeting checks.
4. The current SolidNES run disables spin penalty and S2 observables by design.
   This diagnosis ignores spin, but the lack of observables also removes one
   useful label for identifying which excited root is being optimized.

## Working conclusion

The current gap problem is most likely not caused by the QKV/fused-attention
implementation and not by late state-ordering swaps. It is mainly a convergence
and excited-root-definition problem:

- final-point gaps are not reliable;
- the average gap is closer to the expected Gamma direct-gap scale but still has
  several-eV error bars;
- the second state is only defined through orthogonality and energy minimization,
  without CAS/root-specific pretraining or stronger state labels.

## Next checks

1. Wait for the 15k continuation runs and recompute 1000-step/rolling gap means.
2. Compare overlap matrix, gradient scale, and per-state energy variance across
   the continuation window.
3. After attention is fully stable, add a root-targeted pretraining path or a
   fixed-ground/reference-root workflow; this is the largest method gap relative
   to DeepQMC's excited-state recipe.
4. Use larger/controlled diamond benchmarks only after the two-state root identity
   is stable in the primitive Gamma run.
