# 2026-05-23 FermiNet KFAC Batch4096 Iter400

Added and ran a FermiNet PBC diamond KFAC benchmark using batch size `4096`
and `400` optimization steps.

Reason:

The batch4096 Adam/default run showed that `laplacian: default` is faster than
FOLX for the current 12-electron diamond primitive-cell baseline. This run keeps
that faster Laplacian setting and switches the optimizer to KFAC.

Added configs:

- `configs/train/ground_state_ferminet_pbc_kfac_batch4096_iter400_defaultlap.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_defaultlap.yaml`

Config:

- Optimizer: `kfac`
- Iterations: `400`
- Batch size: `4096`
- Laplacian: `default`
- Learning rate: `0.03`
- KFAC damping: `0.001`
- KFAC covariance update: every step
- KFAC inverse update: every step
- FOLX Forward Laplacian: disabled

Scheduler compliance:

- Submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0038_ferminet_kfac_batch4096_iter400_defaultlap/outputs/slurm_plans/ferminet_kfac_batch4096_iter400_defaultlap_plan.json`.
- Blocked partitions included `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- GPU request: `--gres=gpu:1`
- GPU model: A100 80GB
- Job: `127842`

Run result:

- Start/end: `2026-05-23 13:17:39 CST` to
  `2026-05-23 13:20:48 CST`
- Runtime: `189 s`
- Seconds / step: `0.4725`
- Rows: `400`
- First energy: `-20.792610 Ha`
- Last energy: `-73.746570 Ha`
- Minimum energy: `-73.897995 Ha` at step `387`
- First-50 mean: `-22.187506 Ha`
- Last-50 mean: `-73.443896 Ha`
- Mean `pmove`: `0.777777`
- Last `ewvar`: `0.018472`
- FOLX tile warnings: `0`
- Tracebacks: `0`

Comparison to Adam batch4096/default:

| Metric | Adam job 127840 | KFAC job 127842 |
| --- | ---: | ---: |
| Iterations | 100 | 400 |
| Runtime | 110 s | 189 s |
| Seconds / step | 1.10 | 0.4725 |
| Last energy | -29.266440 Ha | -73.746570 Ha |
| Last-50 mean | -28.578944 Ha | -73.443896 Ha |
| Last `ewvar` | 0.092456 | 0.018472 |
| Mean `pmove` | 0.901304 | 0.777777 |

Assessment:

The KFAC route is now confirmed to run on the latest-JAX FermiNet PBC diamond
framework. For this setup, it reaches a physically plausible energy scale far
faster than the Adam short benchmark and has no warning/traceback issues. The
400-step result is still not a convergence result; it is a strong runtime and
early-optimization pass.

Matched Forward Laplacian follow-up:

- A matching KFAC/FOLX run was submitted afterward as job `127843`; see
  `records/progress/2026-05-23_ferminet_kfac_forward_laplacian_ablation.md`.
- That matched comparison is the appropriate answer to whether Forward
  Laplacian improves speed under KFAC/400/batch4096.
