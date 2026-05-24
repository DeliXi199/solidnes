# Progress

Last updated: 2026-05-24

## Current State

Carbon-diamond benchmark reproduction is complete through both DeepSolid and
FermiNet. The FermiNet PBC-HF pretraining implementation and diamond-Gamma
validation milestone is also complete for the current cc-pVDZ workflow.

The project is now ready to move from ground-state/pretraining route hardening
to reproducing the Szabo and Noe JCTC 2024 penalty-based excited-state VMC
method in code, followed by controlled material tests.

## Active Step

See `ACTIVE_TASK.md` for the exact state and next command.

Short version:

- Active task: Excited-state penalty-VMC method reproduction.
- State: ready_to_start.
- Evidence: GPU target/backend probes `0047--0050`, training integration and
  matched controls `0053--0062`.
- Next action: create the first numbered task bundle under
  `tasks/excited_state_nesvmc/` and implement the first FermiNet PBC
  penalty-state objective scaffold.

## Completed Structural Cleanup

- Legacy generated artifacts were migrated into numbered `tasks/` bundles.
- The retired target-specific scaffold was removed.
- `tasks/TASK_LEDGER.md` is the readable task ledger.
- `records/run_index.md` is the run-number allocator; the next available run
  number is `0063`.
- Top-level `results/`, `outputs/`, and `logs/` are retired and should not be
  recreated for new work.

## Evidence Snapshot

DeepSolid direct route:

- Job `127816`
- Tail-2000 mean: `-75.4161279970 Ha`
- Paper reference: `-75.4009 Ha`

FermiNet route:

- Training job `127898`
- Evaluation job `127992`
- Fixed-parameter evaluation mean: `-75.4125655570 Ha`
- Paper reference: `-75.4009 Ha`

FermiNet PBC-HF pretraining:

- Implementation record:
  `records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md`
- Milestone record:
  `records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md`
- JAX PBC GTO cc-pVDZ validation at image cutoff `3`: AO max abs `1.12e-9`,
  occupied-MO max abs `8.51e-10`.
- JAX PBC GTO cc-pVDZ pretraining target was about `2.52x` faster overall than
  the PySCF target benchmark for the current probe.
- Fixed-iteration early training favored pretraining; short wall-clock
  timeboxed comparisons were mixed, so pretraining is a validated option rather
  than a universal default.

Next-phase task area:

- `tasks/excited_state_nesvmc/`
- Purpose: future numbered task bundles for reproducing the paper-style
  penalty-based excited-state VMC method and testing it on specific periodic
  materials.

## Current Rules

- New artifacts go under one numbered task bundle:
  `tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/`.
- Each task bundle contains `results/`, `outputs/`, and `logs/`.
- SLURM submitters require `SOLIDNES_TASK_ROOT` or explicit plan/log paths.
- Every completed or materially updated task must update
  `tasks/TASK_LEDGER.md`.
- Substantial completed work should also write a dated note under
  `records/progress/`.

## History

Detailed dated history lives in `records/progress/`.

The previous long top-level progress file was archived at:

```text
records/progress/2026-05-23_progress_snapshot_before_top_level_slimming.md
```
