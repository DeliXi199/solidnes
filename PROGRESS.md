# Progress

Last updated: 2026-05-23

## Current State

Carbon-diamond benchmark reproduction is complete through both DeepSolid and
FermiNet. The project is now focused on FermiNet PBC-HF pretraining and, after
that path is robust, the controlled NES-VMC excited-state extension.

## Active Step

See `ACTIVE_TASK.md` for the exact state and next command.

Short version:

- Active task: FermiNet PBC-HF pretraining validation.
- State: ready for dry-run and submission preparation.
- Task bundle:
  `tasks/ferminet_pretraining/0045_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_pbc_hf_pretrain_paper_pilot/`
- Next action: run the FermiNet GPU submitter in dry-run mode with
  `SOLIDNES_TASK_ROOT` set to the task bundle above.

## Completed Structural Cleanup

- Legacy generated artifacts were migrated into numbered `tasks/` bundles.
- The retired target-specific scaffold was removed.
- `tasks/TASK_LEDGER.md` is the readable task ledger.
- `records/run_index.md` is the run-number allocator; the next available run
  number is `0047`.
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
