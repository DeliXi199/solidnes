# 0012 Long-Run Final Checkpoint Policy

Date: 2026-05-29

## Status

Accepted.

## Decision

Every SolidNES iterative training or evaluation task with
`iterations >= 1000` must save a checkpoint for its final training step.

For FermiNet-style `qmcjax` checkpoint naming, an `N`-iteration run is complete
only when the checkpoint directory contains:

```text
qmcjax_ckpt_{N-1}.npz
```

Intermediate time-based checkpoints are useful for preemption recovery, but
they are not enough to mark a long task complete.

## Rationale

Several long runs are compared after queue delays and GPU scheduling changes.
If a run stops at the requested iteration count but only has an older
time-based checkpoint, continuation, evaluation, and method comparisons can
silently start from the wrong optimization state.

## Consequences

- FermiNet/PsiFormer runs launched through `scripts/backends/run_ferminet_train.py`
  enforce the final-step checkpoint and fail if the expected final checkpoint
  is absent.
- DeepSolid configs built through the SolidNES adapter set step-based
  checkpointing for long runs when `checkpoint_every_steps` is not explicitly
  configured.
- Task summaries should report the final checkpoint path for long runs.
