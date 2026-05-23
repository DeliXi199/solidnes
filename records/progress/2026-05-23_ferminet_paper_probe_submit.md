# 2026-05-23 FermiNet Paper-Like Probe Submission

Submitted the first paper-like FermiNet PBC diamond comparison probe.

Purpose:

Run a heavier FermiNet configuration before launching the full 10000-step
FermiNet-vs-DeepSolid comparison. This probe measures real throughput with the
larger model, MCMC20, KFAC, FOLX, and the full `intelgpu80g/gpu001` node.

Added configs:

- `configs/model/ferminet_pbc_paper.yaml`
- `configs/sampler/metropolis_ferminet_pbc_mcmc20.yaml`
- `configs/train/ground_state_ferminet_pbc_kfac_folx_batch4096_iter1000.yaml`
- `configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter1000_paper.yaml`

Experiment:

- Name: `diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_iter1000_paper`
- Model: FermiNet, 8 determinants, hidden dims `(256, 32) x 4`
- Optimizer: `kfac`
- Laplacian: `folx`
- Batch size: `4096`
- Iterations: `1000`
- MCMC burn-in: `100`
- MCMC steps per iteration: `20`
- Precision: FP64 disabled

Scheduler compliance:

- Submitted through `scripts/slurm/submit_ferminet_gpu_smoke.sh`.
- Dry-run plan stored at
  `tasks/phase1_diamond_c/pbc_gamma/training/0042_ferminet_kfac_folx_batch4096_iter1000_paper/outputs/slurm_plans/ferminet_paper_kfac_folx_b4096_iter1000_g2_plan.json`.
- Blocked partitions included `test`.
- Hard GPU request: `2`
- Allowed GPU counts: `2`
- CPU request: `96`
- Selected partition/node: `intelgpu80g/gpu001`
- SLURM request: `--gres=gpu:2 --cpus-per-task=96 --exclusive`
- Job: `127847`

Startup verification:

- `CUDA_VISIBLE_DEVICES=0,1`
- `SOLIDNES_EFFECTIVE_CPUS=96`
- `jax_default_backend=gpu`
- `jax_devices=[CudaDevice(id=0), CudaDevice(id=1)]`
- Start time: `2026-05-23 14:21:34 CST`

Pending:

- Wait for job completion.
- Generate benchmark summary.
- Compare throughput and energy against the small-model FOLX KFAC runs and
  existing DeepSolid C-diamond long-run reference.
