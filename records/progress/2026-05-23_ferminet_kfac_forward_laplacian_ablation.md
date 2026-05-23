# 2026-05-23 FermiNet KFAC Forward Laplacian Ablation

Ran the missing KFAC/FOLX comparison for the Forward Laplacian speed test.

Reason:

The earlier KFAC run used `laplacian: default` only. To answer whether Forward
Laplacian improves speed after switching optimizer to KFAC and increasing the
run length to 400 steps, a matched FOLX run was required.

Added configs:

- `configs/train/ground_state_ferminet_pbc_kfac_batch4096_iter400_folx.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter400_folx.yaml`

Matched controls:

- Optimizer: `kfac`
- Iterations: `400`
- Batch size: `4096`
- Network: small FermiNet, 8 determinants, hidden dims `(128, 32) x 3`
- Sampler: burn-in 50, MCMC10
- Precision: FP64 disabled
- GPU: one A100 80GB

Scheduler compliance:

- Submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0039_ferminet_kfac_batch4096_iter400_folx/outputs/slurm_plans/ferminet_kfac_batch4096_iter400_folx_plan.json`.
- Blocked partitions included `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- GPU request: `--gres=gpu:1`
- FOLX job: `127843`

Results:

| Metric | Default job 127842 | FOLX job 127843 |
| --- | ---: | ---: |
| Optimizer | `kfac` | `kfac` |
| Batch size | 4096 | 4096 |
| Iterations | 400 | 400 |
| Laplacian | `default` | `folx` |
| Forward Laplacian enabled | false | true |
| Runtime | 189 s | 198 s |
| Seconds / step | 0.4725 | 0.4950 |
| FOLX / default runtime | 1.00x | 1.0476x |
| FOLX tile warnings | 0 | 0 |
| Tracebacks | 0 | 0 |
| Last energy | -73.746570 Ha | -73.700320 Ha |
| Minimum energy | -73.897995 Ha | -76.323910 Ha |
| Last-50 mean | -73.443896 Ha | -73.488867 Ha |
| Mean `pmove` | 0.777777 | 0.777107 |
| Last `ewvar` | 0.018472 | 0.054945 |

Assessment:

For the matched KFAC, batch4096, 400-step C-diamond primitive-cell benchmark,
Forward Laplacian did not improve speed. The FOLX run was slightly slower:
`0.4950 / 0.4725 = 1.0476x` the default Laplacian runtime, or about `4.8%`
slower end-to-end.

This is a narrower gap than the Adam batch4096 comparison (`1.41x` slower), but
the practical conclusion for the current 12-electron small-cell baseline is the
same: use `laplacian: default` for efficiency unless a larger-electron scaling
benchmark shows a crossover.

Superseding longer-run result:

- A 2000-step matched KFAC comparison was run afterward; see
  `records/progress/2026-05-23_ferminet_kfac_iter2000_forward_laplacian_ablation.md`.
- In the longer run, FOLX became faster after initialization/JIT overhead was
  amortized. Treat this 400-step result as a short-run overhead measurement, not
  the final efficiency conclusion.
