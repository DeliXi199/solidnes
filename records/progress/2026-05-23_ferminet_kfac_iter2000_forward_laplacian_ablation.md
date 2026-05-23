# 2026-05-23 FermiNet KFAC Iter2000 Forward Laplacian Ablation

Ran the longer KFAC/batch4096 Forward Laplacian comparison with `2000`
optimization steps to reduce initialization and compilation overhead.

Reason:

The matched KFAC/batch4096/400-step comparison showed FOLX was `4.8%` slower,
but that run still mixed a large one-time initialization/JIT cost into a short
benchmark. The 2000-step comparison tests whether FOLX improves steady-state
throughput after this overhead is amortized.

Added configs:

- `configs/train/ground_state_ferminet_pbc_kfac_batch4096_iter2000_defaultlap.yaml`
- `configs/train/ground_state_ferminet_pbc_kfac_batch4096_iter2000_folx.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_defaultlap.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_batch4096_iter2000_folx.yaml`

Matched controls:

- Optimizer: `kfac`
- Iterations: `2000`
- Batch size: `4096`
- Network: small FermiNet, 8 determinants, hidden dims `(128, 32) x 3`
- Sampler: burn-in 50, MCMC10
- Precision: FP64 disabled
- GPU request: one A100 80GB per job

Scheduler compliance:

- Both jobs were submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plans blocked `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- Default job: `127844`
- FOLX job: `127845`

Results:

| Metric | Default job 127844 | FOLX job 127845 |
| --- | ---: | ---: |
| Optimizer | `kfac` | `kfac` |
| Batch size | 4096 | 4096 |
| Iterations | 2000 | 2000 |
| Laplacian | `default` | `folx` |
| Forward Laplacian enabled | false | true |
| Runtime | 557 s | 394 s |
| Seconds / step | 0.2785 | 0.1970 |
| Default / FOLX speed ratio | 1.00x | 1.4137x faster |
| FOLX tile warnings | 0 | 0 |
| Tracebacks | 0 | 0 |
| Last energy | -74.983570 Ha | -75.177060 Ha |
| Last-50 mean | -75.020628 Ha | -75.111432 Ha |
| Tail mean, last 1000 rows | -74.931896 Ha | -75.032381 Ha |
| Mean `pmove` | 0.586968 | 0.587727 |
| Last `ewvar` | 0.002558 | 0.002363 |

Comparison across benchmark lengths:

| Benchmark | Default s/step | FOLX s/step | FOLX / default |
| --- | ---: | ---: | ---: |
| Adam, batch4096, 100 steps | 1.10 | 1.55 | 1.41x slower |
| KFAC, batch4096, 400 steps | 0.4725 | 0.4950 | 1.05x slower |
| KFAC, batch4096, 2000 steps | 0.2785 | 0.1970 | 0.71x runtime |

Two-point overhead estimate from KFAC 400 and 2000:

- Default: approximate steady-state slope `(557 - 189) / 1600 = 0.230 s/step`
  with about `97 s` fixed overhead.
- FOLX: approximate steady-state slope `(394 - 198) / 1600 = 0.1225 s/step`
  with about `149 s` fixed overhead.

Assessment:

The 2000-step benchmark supports the initialization-overhead hypothesis. FOLX
has higher fixed overhead, but after amortization it is faster for the current
KFAC/batch4096 setup. For long KFAC runs on this FermiNet diamond baseline,
Forward Laplacian should be enabled. For very short smoke tests, default
Laplacian can still look faster because it pays less one-time overhead.
