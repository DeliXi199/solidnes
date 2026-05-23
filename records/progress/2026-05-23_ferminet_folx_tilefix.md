# 2026-05-23 FermiNet FOLX Tile Fix

Removed the repeated FOLX warning from the FermiNet PBC diamond short baseline.

Change:

- Local file changed: `external/ferminet/ferminet/networks.py`
- Upstream clone HEAD: `c4312c315dda1c5728994ba89629744f71c6eb66`
- Reproducible patch stored at:
  `patches/ferminet/folx_tile_broadcast.patch`
- Replacement:
  `jnp.tile(g, [h_one.shape[0], 1])` to
  `jnp.broadcast_to(g, (h_one.shape[0], g.shape[-1]))`

Reason:

FOLX warned that `tile` was not in its registry during FermiNet symmetric
feature construction. This can force a less efficient fallback path inside the
Forward Laplacian transform. Broadcasting expresses the same repeated view
without materializing a tiled array and avoids that FOLX primitive warning.

Scheduler compliance:

- Submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0034_ferminet_adam_short100_folx_tilefix/outputs/slurm_plans/ferminet_adam_short100_folx_tilefix_plan.json`.
- Blocked partitions included `test`.
- Selected partition/node: `intelgpu80g/gpu001`
- GPU request: `--gres=gpu:1`
- GPU model: A100 80GB
- Job: `127833`

Run:

- Experiment: `diamond_c_ferminet_pbc_gamma_adam_short100_folx_tilefix`
- Start/end: `2026-05-23 12:49:21 CST` to
  `2026-05-23 12:51:18 CST`
- Runtime: `117 s`, `1.17 s/optimization step`
- Rows: `100`
- First energy: `-19.545017 Ha`
- Last energy: `-23.293020 Ha`
- Minimum energy: `-27.982136 Ha` at step `65`
- First-50 mean: `-22.971573 Ha`
- Last-50 mean: `-25.811617 Ha`
- Mean `pmove`: `0.903578`
- Last `ewvar`: `1.103648`

Before/after comparison:

| Metric | Before job 127830 | After job 127833 |
| --- | ---: | ---: |
| FOLX `tile` warnings | 18 | 0 |
| Tracebacks | 0 | 0 |
| Runtime | 117 s | 117 s |
| Seconds / step | 1.17 | 1.17 |
| Last-50 energy mean | -25.320289 Ha | -25.811617 Ha |
| Mean `pmove` | 0.901219 | 0.903578 |

Assessment:

The warning is fixed without changing the scheduler path or increasing runtime
at this small scale. The energy differences are not interpreted as a physics
improvement because both runs are short stochastic trend checks with different
random trajectories.
