# 2026-05-23 FermiNet Forward Laplacian Ablation

Compared FermiNet PBC diamond short runs with and without FOLX Forward
Laplacian.

Purpose:

- Check whether `cfg.optim.laplacian = "folx"` is faster than FermiNet's
  default Laplacian path for the current small diamond Gamma-point baseline.
- Keep all other controls aligned with the 100-step Adam short baseline.

Configs:

- FOLX: `diamond_c_ferminet_pbc_gamma_adam_short100_folx_tilefix`
- Default Laplacian: `diamond_c_ferminet_pbc_gamma_adam_short100_defaultlap`
- New no-FOLX train config:
  `configs/train/ground_state_ferminet_pbc_adam_short100_defaultlap.yaml`
- New no-FOLX experiment config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100_defaultlap.yaml`

Scheduler compliance:

- The no-FOLX comparison was submitted through
  `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0035_ferminet_adam_short100_defaultlap/outputs/slurm_plans/ferminet_adam_short100_defaultlap_plan.json`.
- Blocked partitions included `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- GPU request: `--gres=gpu:1`
- GPU model: A100 80GB
- Job: `127837`

Results:

| Metric | FOLX job 127833 | Default job 127837 |
| --- | ---: | ---: |
| Laplacian | `folx` | `default` |
| Forward Laplacian enabled | true | false |
| Rows | 100 | 100 |
| Runtime | 117 s | 57 s |
| Seconds / step | 1.17 | 0.57 |
| FOLX tile warnings | 0 | 0 |
| Tracebacks | 0 | 0 |
| Mean `pmove` | 0.903578 | 0.899469 |
| Last-50 energy mean | -25.811617 Ha | -26.592552 Ha |

Interpretation:

For this small 12-electron diamond primitive-cell, 100-step, batch-64 run,
FOLX is slower end-to-end: `1.17 / 0.57 = 2.05x` the default Laplacian runtime.
This includes startup and compilation overhead, so it is a practical small-run
benchmark rather than a pure steady-state kernel benchmark.

This does not invalidate Forward Laplacian for the project. It means the
current FOLX path should not be assumed to be faster at the smallest baseline
size. The next meaningful test is a scaling benchmark with a larger electron
count, larger batch/model, or a longer run where compilation overhead is less
dominant.
