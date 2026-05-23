# 2026-05-23 FermiNet Batch4096 Ablation

Increased the FermiNet PBC diamond Adam short benchmark batch size from `64`
to `4096` and reran the FOLX/default Laplacian comparison.

Reason:

The first Forward Laplacian ablation used batch size `64`, which is too small
to make good use of an A100 and can overstate transform/compilation overhead.
The batch4096 run tests whether FOLX improves once the per-step work is larger.

Added configs:

- `configs/train/ground_state_ferminet_pbc_adam_short100_batch4096.yaml`
- `configs/train/ground_state_ferminet_pbc_adam_short100_batch4096_defaultlap.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_folx.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_adam_short100_batch4096_defaultlap.yaml`

Scheduler compliance:

- Both jobs were submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plans blocked `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- GPU request: `--gres=gpu:1` for each job
- GPU model: A100 80GB
- Default job: `127840`
- FOLX job: `127841`

Batch4096 results:

| Metric | Default job 127840 | FOLX job 127841 |
| --- | ---: | ---: |
| Batch size | 4096 | 4096 |
| Laplacian | `default` | `folx` |
| Forward Laplacian enabled | false | true |
| Rows | 100 | 100 |
| Runtime | 110 s | 155 s |
| Seconds / step | 1.10 | 1.55 |
| FOLX tile warnings | 0 | 0 |
| Tracebacks | 0 | 0 |
| Mean `pmove` | 0.901304 | 0.900885 |
| Last-50 energy mean | -28.578944 Ha | -28.696910 Ha |
| Last `ewvar` | 0.092456 | 0.076433 |

Comparison to batch64:

| Batch | Default s/step | FOLX s/step | FOLX / default |
| ---: | ---: | ---: | ---: |
| 64 | 0.57 | 1.17 | 2.05x |
| 4096 | 1.10 | 1.55 | 1.41x |

Assessment:

Increasing batch size from `64` to `4096` narrows the FOLX overhead, but FOLX
is still slower for the current 12-electron small FermiNet diamond primitive
cell: `1.55 / 1.10 = 1.41x` the default Laplacian runtime.

This suggests the current efficiency baseline should use `laplacian: default`
for this small C-diamond setup. Forward Laplacian should be retested on larger
electron counts or larger networks where Hessian-style Laplacian cost dominates
more strongly.
