# Long-Run Final Checkpoint Policy

Date: 2026-05-29

## Change

Added the SolidNES rule that any iterative job with `iterations >= 1000` must
save its final-step checkpoint.

## Implementation

- Documented the rule in `PROJECT_GUIDE.md` and SLURM task-management safety
  rules.
- Added decision record `0012_long_run_final_checkpoint_policy.md`.
- Added FermiNet/PsiFormer runtime enforcement in `run_ferminet_train.py`.
- Added DeepSolid adapter support for final-step checkpointing through
  `save_frequency_in_step` when the training config does not explicitly set
  `checkpoint_every_steps`.

## Operational Meaning

For an `N`-iteration FermiNet-style run, the long task is complete only when
`qmcjax_ckpt_{N-1}.npz` exists in the run checkpoint directory.
