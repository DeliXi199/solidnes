# 2026-05-23 FermiNet Paper-Like 10000-Step Submission

Submitted the formal paper-like FermiNet PBC diamond run for comparison against
the existing DeepSolid C-diamond long-run reference.

Added configs:

- `configs/train/ground_state_ferminet_pbc_kfac_folx_batch4096_iter10000.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter10000_paper.yaml`

Experiment:

- Name: `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter10000_paper`
- Model: FermiNet, 8 determinants, hidden dims `(256, 32) x 4`
- Optimizer: `kfac`
- Laplacian: `folx`
- Batch size: `4096`
- Iterations: `10000`
- MCMC burn-in: `100`
- MCMC steps per iteration: `20`
- Precision: FP64 disabled

Scheduler compliance:

- Submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0043_ferminet_kfac_folx_batch4096_iter10000_paper/outputs/slurm_plans/ferminet_paper_kfac_folx_b4096_iter10000_g2_plan.json`.
- Blocked partitions included `test`.
- Hard GPU request: `2`
- Allowed GPU counts: `2`
- CPU request: `96`
- Selected partition/node: `intelgpu80g/gpu001`
- SLURM request: `--gres=gpu:2 --cpus-per-task=96 --exclusive`
- Job: `127848`

Startup verification:

- `CUDA_VISIBLE_DEVICES=0,1`
- `SOLIDNES_EFFECTIVE_CPUS=96`
- `jax_default_backend=gpu`
- `jax_devices=[CudaDevice(id=0), CudaDevice(id=1)]`
- Start time: `2026-05-23 14:37:33 CST`

Runtime expectation:

- The prior 1000-step probe job `127847` took `428 s`, or `0.428 s/step`.
- Linear extrapolation gives about `71 min` for 10000 steps.
- Walltime was set to `04:00:00` to leave room for JIT/KFAC overhead and
  runtime variability.

## Completed Result

Job `127848` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 00:29:16 by sacct
log runtime: 2026-05-23 14:37:33 to 15:06:41 CST, 1748 s
node: gpu001
resources: 2 GPU on intelgpu80g, 96 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/pbc_gamma/training/0043_ferminet_kfac_folx_batch4096_iter10000_paper/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/pbc_gamma/training/0043_ferminet_kfac_folx_batch4096_iter10000_paper/results/checkpoints/qmcjax_ckpt_005935.npz
tasks/phase1_diamond_c/pbc_gamma/training/0043_ferminet_kfac_folx_batch4096_iter10000_paper/results/validation/benchmark_summary.json
tasks/phase1_diamond_c/pbc_gamma/training/0043_ferminet_kfac_folx_batch4096_iter10000_paper/results/validation/benchmark_summary.md
```

Main statistics:

```text
rows: 10000
first step: 0
last step: 9999
first energy: -18.707180 Ha
last energy: -75.281360 Ha
minimum energy: -78.686110 Ha at step 511
last-10 mean: -75.2650287 Ha
last-50 mean: -75.26011696 Ha
tail mean, last 5000 rows: -75.2595080136 Ha
tail block stderr, 5 blocks: 0.0027959247 Ha
mean pmove: 0.5538866356
tail pmove mean: 0.5358442845
last ewvar: 0.0017481738
seconds per optimization step: 0.1748
FOLX tile warnings: 0
tracebacks: 0
```

The five tail block means are:

```text
-75.2508841240
-75.2578070400
-75.2588885480
-75.2619137300
-75.2680466260 Ha
```

## Comparison With DeepSolid 150000-Step Reference

Reference:
`records/progress/2026-05-22_carbon_diamond_ccpvdz_continue_150000.md`

DeepSolid reference job `122085` used the same `intelgpu80g/gpu001` node class
with `2 GPU + 96 CPU`, model width `(256, 32) x 4`, 8 determinants, batch
`384`, MCMC20, Adam, and cc-pVDZ. It completed in `05:50:50`.

Key comparison:

```text
FermiNet 10k elapsed: 1748 s
DeepSolid 150k continuation elapsed: 21050 s
wall-clock ratio: 12.04x shorter for FermiNet

FermiNet last energy: -75.2813600000 Ha
DeepSolid last energy: -75.2704621344 Ha
difference: -0.0108978656 Ha

FermiNet tail mean, last 5000 rows: -75.2595080136 Ha
DeepSolid tail mean, last 50000 rows: -75.1982415149 Ha
difference: -0.0612664987 Ha

FermiNet last-50 mean: -75.2601169600 Ha
DeepSolid final tail block: -75.2044956447 Ha
difference: -0.0556213153 Ha
```

Interpretation:

The 10000-step FermiNet KFAC/FOLX run reached a lower robust tail estimate than
the existing 150000-step DeepSolid continuation while using about one twelfth
of the wall-clock time on the same 2-GPU A100 node. This is not a perfect
apples-to-apples optimizer and ansatz comparison because the DeepSolid
reference uses Adam and the cc-pVDZ DeepSolid setup, but it is a strong
project-level signal that the FermiNet route is the better efficiency baseline
for the next stage.
