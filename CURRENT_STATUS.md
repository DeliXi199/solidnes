# SolidNES Current Status

Last updated: 2026-05-23, Asia/Shanghai

## Current Conclusion

The carbon-diamond paper benchmark milestone is complete at the project
validation level.

Both routes now reproduce the DeepSolid supplementary diamond VMC reference
energy and reach equal or lower energies:

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
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/evaluation_trace.png
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/evaluation_vs_training.png
tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/
```

## Current Task State

There are no active Slurm jobs for the current user at the time of this status
update.

```text
squeue -u $USER returned only the header row.
```

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
9. Initial true PySCF PBC-HF pretraining support for the FermiNet PBC route was
   added and passed build/local CPU probes.
10. Numbered task bundles, `tasks/TASK_LEDGER.md`, and the retired top-level
    generated artifact cleanup are in place.

### Current Milestone

```text
Diamond ground-state benchmark reproduction is complete for both:
- DeepSolid direct calculation
- FermiNet calculation
```

The project can now move from "can we reproduce the paper energy?" to
"can we make the FermiNet route faster, cleaner, and more systematic?"

## Important Caveats

1. DeepSolid job `127816` is a training-chain statistic, not yet a separate
   fixed-checkpoint evaluation. It is already strong enough for the current
   milestone, but a fixed-checkpoint DeepSolid evaluation would make the
   comparison cleaner.
2. FermiNet evaluation job `127992` restored `qmcjax_ckpt_018349.npz`, because
   the 20k training job did not write a final `qmcjax_ckpt_019999.npz`.
3. The current FermiNet result is already strong, but a longer fixed evaluation
   can reduce the statistical error bar if needed for a formal report.
4. The retired target-specific scaffold has been removed; the current milestone
   is carbon diamond and the FermiNet PBC-HF pretraining path.

## Next Phase

The next main task is to improve and validate the FermiNet PBC-HF pretraining
path.

Current implementation status:

```text
PBC-HF pretraining code exists.
Build-only checks pass.
Paper-geometry cc-pVDZ PySCF PBC-HF target converged.
A minimal local CPU one-step PBC pretraining probe passed with a sto-3g target.
```

Next work items:

1. Run a short GPU PBC-HF pretraining pilot through the approved Slurm GPU
   submitter.
2. Add clearer pretraining diagnostics: pretrain loss, HF target shape, walker
   refresh behavior, checkpoint path, and whether the final pretrained
   parameters are restored by VMC training.
3. Compare matched runs:

```text
No pretraining x64 baseline:
  FermiNet x64 KFAC/FOLX batch4096 MCMC20

PBC-HF pretraining run:
  same geometry, same model, same batch, same KFAC/FOLX settings
```

4. Use short comparison windows first, such as 2000 or 5000 VMC steps, to check
   whether pretraining improves early energy and variance.
5. If useful, launch a longer pretrained production run and then a
   fixed-parameter evaluation.
6. Keep all GPU submissions compliant with the project scheduling rule:

```text
Use scripts/slurm/submit_ferminet_gpu_smoke.sh
Run dry-run first
Do not submit GPU compute to the test partition
Respect node/GPU/CPU allocation rules
```

## Working Decision

The working decision is:

```text
Use DeepSolid as the successful direct-paper-reproduction reference.
Use FermiNet as the primary efficiency and future-development framework.
Focus next on making FermiNet PBC-HF pretraining robust and measurable.
```
