# SolidNES Current Status

Last updated: 2026-05-24, Asia/Shanghai

## Current Conclusion

Two Phase 1 enabling milestones are complete at the project-validation level:

1. The carbon-diamond paper benchmark reproduction is complete through both
   DeepSolid and FermiNet.
2. The FermiNet PBC-HF pretraining implementation and diamond-Gamma validation
   milestone is complete for the current cc-pVDZ workflow.

The project can now move from ground-state and pretraining route hardening to
the next phase: reproduce the Szabo and Noe JCTC 2024 penalty-based
excited-state VMC method in the SolidNES code path, then test it on concrete
periodic materials.

## Carbon-Diamond Benchmark

Both routes reproduce the DeepSolid supplementary diamond VMC reference energy
and reach equal or lower energies:

```text
DeepSolid supplementary diamond VMC reference: -75.4009 Ha
```

### DeepSolid Direct Route

The direct DeepSolid calculation reached the paper result.

Evidence:

```text
Job ID: 127816
Backend: DeepSolid
Setup: C diamond, cc-pVDZ, KFAC, batch4096, MCMC20
Resources: 4 GPU + 64 CPU
Elapsed: 03:30:43
Rows: 10000, steps 1000--10999
Last energy: -75.4010653122 Ha
Last-100 mean: -75.4192298683 Ha
Tail-2000 mean: -75.4161279970 Ha
Tail block stderr: 0.0006453174 Ha
Mean pmove: 0.5397935999
Final variance: 0.5978799387
```

Interpretation:

```text
The last energy already matches/slightly beats -75.4009 Ha, and the robust
tail-2000 mean is lower by about 15.23 mHa. This is enough to count the
DeepSolid direct route as a successful benchmark reproduction.
```

Record:

```text
records/progress/2026-05-22_carbon_diamond_ccpvdz_paper_kfac_probe.md
tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/validation/training_summary.md
```

### FermiNet Route

The FermiNet x64 route reached and exceeded the same paper result, with an
independent fixed-parameter evaluation.

Evidence:

```text
Training job: 127898
Evaluation job: 127992
Backend: FermiNet
Setup: C diamond paper geometry, x64, KFAC, FOLX Forward Laplacian,
       batch4096, MCMC20
Evaluation checkpoint: qmcjax_ckpt_018349.npz
Evaluation resources: 2 GPU + 96 CPU
Evaluation elapsed: 00:22:29
Evaluation rows: 2000
Evaluation all-row mean: -75.4125655570 Ha
Evaluation 5-block stderr: 0.0004411545 Ha
Evaluation last-1000 mean: -75.4118625314 Ha
Evaluation last energy: -75.4374770841 Ha
Mean pmove: 0.5362353088
Last ewvar: 0.0001866115
FOLX tile warnings: 0
Tracebacks: 0
```

Interpretation:

```text
The fixed-parameter evaluation mean is lower than -75.4009 Ha by about
11.67 mHa. It also agrees with the x64 training tail-10000 mean
(-75.4114059555 Ha) within about 1.16 mHa, so the FermiNet result is not just
a transient training-chain value.
```

Records:

```text
records/progress/2026-05-23_ferminet_x64_paper_pilot20k_submit.md
records/progress/2026-05-23_ferminet_x64_eval_ckpt18349_iter2000_submit.md
records/progress/2026-05-23_task_bundle_migration.md
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/benchmark_summary.md
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/
```

## FermiNet PBC-HF Pretraining

The FermiNet PBC-HF pretraining route is implemented, GPU-tested, and usable
for the current carbon-diamond Gamma, cc-pVDZ workflow.

Evidence:

```text
Implementation record: records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
Milestone record: records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md

PySCF PBC-HF target benchmark:
  Run 0047
  Loss: 2.14205 -> 0.0179985
  Steady mean step: 0.058384 s

JAX PBC GTO cc-pVDZ validation:
  Run 0050
  Image cutoff: 3
  AO max abs error: 1.12e-9
  Occupied-MO max abs error: 8.51e-10
  Mean step: 0.023191 s
  Speedup vs PySCF cc-pVDZ target benchmark: about 2.52x total step

Training integration:
  Runs 0053, 0054: pretrain + KFAC/FOLX completed without traceback
  Runs 0055, 0056: matched no-pretrain 1000-step controls
  Runs 0057, 0058: timeboxed no-pretrain controls
  Runs 0059--0062: timeboxed pretrain-length sweep
```

Interpretation:

```text
PBC-HF pretraining is now a working, instrumented FermiNet option for diamond
Gamma cc-pVDZ. It improves fixed-iteration early training in the 1000-step
comparison, but the matched short wall-clock comparisons are mixed. It should
be kept as a validated option, not treated yet as a universal production
default.
```

## Current Task State

There are no active Slurm jobs for the current user at the time of this status
update.

The excited-state penalty-VMC route now has backend-independent overlap and
penalty utilities, a reusable FermiNet PBC external-state adapter, and
build-only cheap-local-energy checks for penalty terms, one gradient step, and
three consecutive SGD updates. The next unproven path is a controlled real PBC
local-energy/Laplacian smoke.

## Milestones

### Completed

1. Project scaffold and task records are in place.
2. DeepSolid environment and GPU execution path were made usable on modern JAX.
3. Carbon-diamond DeepSolid smoke, continuation, and validation runs completed.
4. Direct DeepSolid paper-like KFAC/cc-pVDZ run reached the paper diamond VMC
   reference.
5. FermiNet PBC diamond scaffold was built on latest JAX.
6. FOLX Forward Laplacian was enabled and benchmarked; long KFAC runs showed
   clear speed benefit after overhead is amortized.
7. FermiNet x64 paper-geometry run reached the paper diamond VMC reference.
8. FermiNet fixed-parameter evaluation confirmed the x64 training result.
9. True PBC-HF pretraining support for the FermiNet PBC route was added.
10. JAX PBC GTO pretraining target evaluation was validated for diamond Gamma
    `ccpvdz` and benchmarked faster than the host-side PySCF target path.
11. FermiNet PBC-HF pretraining completed GPU integration checks and matched
    no-pretrain comparisons for the current diamond Gamma workflow.
12. Numbered task bundles, `tasks/TASK_LEDGER.md`, and the retired top-level
    generated artifact cleanup are in place.

### Current Milestone

```text
Completed:
- Diamond ground-state benchmark reproduction for both:
  - DeepSolid direct calculation
  - FermiNet calculation
- FermiNet PBC-HF pretraining implementation and diamond-Gamma validation
```

The next milestone is the first controlled two-state periodic NES-VMC probe.

## Important Caveats

1. DeepSolid job `127816` is a training-chain statistic, not yet a separate
   fixed-checkpoint evaluation. It is already strong enough for the current
   milestone, but a fixed-checkpoint DeepSolid evaluation would make the
   comparison cleaner.
2. FermiNet evaluation job `127992` restored `qmcjax_ckpt_018349.npz`, because
   the 20k training job did not write a final `qmcjax_ckpt_019999.npz`.
3. The current FermiNet result is already strong, but a longer fixed evaluation
   can reduce the statistical error bar if needed for a formal report.
4. PBC-HF pretraining is validated for the current diamond Gamma `ccpvdz`
   workflow only. Non-Gamma twists, other cells, pseudopotentials, or larger
   basis sets still require AO/MO validation.
5. PBC-HF pretraining improves fixed-iteration early training in the tested
   1000-step comparison, but short wall-clock timeboxed comparisons are mixed.
   It is not yet a universal production default.

## Next Phase

The next main task is excited-state method reproduction and material testing.

Recommended next work items:

1. Reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC method
   in code, starting with the FermiNet PBC backend.
2. Keep the first target narrow: carbon diamond primitive cell, Gamma point,
   same ground-state geometry and basis, and clear orthogonality/overlap
   diagnostics.
3. Use `tasks/excited_state_nesvmc/` for future numbered implementation,
   smoke, training, evaluation, and material-test task bundles only when those
   steps produce project artifacts. Pure source audits and design notes should
   stay in `docs/` or `records/progress/`.
4. After the controlled diamond probe works, choose concrete material tests and
   record direct-gap, indirect-gap, twist, and finite-size caveats explicitly.
5. Use PBC-HF pretraining as an optional initialization path, not a mandatory
   default, until longer wall-clock evidence justifies it.
6. Preserve DeepSolid as the successful ground-state reproduction reference and
   FermiNet as the primary development framework.

## Working Decision

The working decision is:

```text
Use DeepSolid as the successful direct-paper-reproduction reference.
Use FermiNet as the primary efficiency and future-development framework.
Treat FermiNet PBC-HF pretraining as a completed enabling milestone for diamond
Gamma cc-pVDZ, with mixed wall-clock conclusions.
Focus next on reproducing the penalty-based excited-state VMC method in code
and then testing it on concrete periodic materials.
```
