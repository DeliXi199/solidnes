# 2026-05-23 FermiNet X64 Fixed-Checkpoint Evaluation 2000-Step Submission

Created and submitted a quick fixed-parameter evaluation pilot from the x64
paper-geometry FermiNet training checkpoint.

Purpose:

```text
Check whether the trained wavefunction still estimates around -75.40 Ha when
the network parameters are fixed and only MCMC sampling/local-energy evaluation
is performed.
```

Added configs:

```text
configs/sampler/metropolis_ferminet_pbc_evaluation_mcmc20.yaml
configs/train/ground_state_ferminet_pbc_eval_none_batch4096_iter2000.yaml
configs/experiment/diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000.yaml
```

Configuration:

```text
experiment: diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000
restore directory: tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints
restore checkpoint selected by FermiNet: qmcjax_ckpt_018349.npz
optimizer: none
iterations: 2000
batch size: 4096
MCMC burn-in: 0
MCMC steps per iteration: 20
precision: x64 enabled
laplacian: FOLX
```

Caveat:

```text
The 20k training job did not write qmcjax_ckpt_019999.npz at the end. The
latest valid saved checkpoint is qmcjax_ckpt_018349.npz, so this evaluation
uses that checkpoint rather than the exact final training parameters.
```

Build-only verification passed:

```text
optimizer: none
iterations: 2000
batch_size: 4096
forward_laplacian_enabled: True
mcmc_burn_in: 0
mcmc_steps_per_iteration: 20
precision_profile: fp64
x64_enabled: True
restore_path: /data/home/yihaoxu/research/projects/solidnes/tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints
```

Scheduler compliance:

```text
submitter: scripts/slurm/submit_ferminet_gpu_smoke.sh
dry-run plan: tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/outputs/slurm_plans/ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000_g2_plan.json
blocked partitions: includes test
selected partition/node: intelgpu80g/gpu001
request: --gres=gpu:2 --cpus-per-task=96 --exclusive
time limit: 02:00:00
precision scheduling profile: fp64
```

Submitted job:

```text
job id: 127992
job name: solidnes-ferminet-eval2k
state after submission: RUNNING
node: gpu001
CUDA_VISIBLE_DEVICES: 0,1
JAX_ENABLE_X64: 1
jax: 0.10.1
jaxlib: 0.10.1
jax_default_backend: gpu
jax_devices: [CudaDevice(id=0), CudaDevice(id=1)]
start time: 2026-05-23 19:33:31 CST
```

Follow-up after completion:

```text
python scripts/validation/summarize_ferminet_benchmark.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_x64_eval_ckpt18349_batch4096_mcmc20_iter2000.yaml \
  --job-id 127992 \
  --log tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/logs/slurm/solidnes-ferminet-eval2k_127992.log \
  --err tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/logs/slurm/solidnes-ferminet-eval2k_127992.err \
  --plan-json tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/outputs/slurm_plans/ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000_g2_plan.json
```

Completion:

```text
SLURM state: COMPLETED 0:0
elapsed: 00:22:29
allocation: intelgpu80g/gpu001, 2 GPU, 96 CPU
rows: 2000
first/last step: 0 / 1999
FOLX tile warnings: 0
tracebacks: 0
```

Evaluation result:

```text
all-row energy mean: -75.4125655570 Ha
5-block stderr: 0.0004411545 Ha
last-1000 mean: -75.4118625314 Ha
last-500 mean: -75.4115203093 Ha
last-100 mean: -75.4121985065 Ha
last energy: -75.4374770841 Ha
EW mean at last row: -75.4170163351 Ha
mean pmove: 0.5362353088
last ewvar: 0.0001866115
DeepSolid SI diamond reference: -75.4009 Ha
all-row mean minus reference: -0.0116655570 Ha
last-1000 mean minus reference: -0.0109625314 Ha
```

Interpretation:

```text
The 2000-step fixed-parameter evaluation is consistent with the x64 training
tail. The all-row evaluation mean is only -0.0011596015 Ha lower than the
training tail-10000 mean (-75.4114059555 Ha), so the previous x64 result is not
just a transient training-chain artifact. This is still a short evaluation and
uses qmcjax_ckpt_018349.npz because the 20k training run did not write a final
step-19999 checkpoint.
```

Generated artifacts:

```text
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/benchmark_summary.json
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/benchmark_summary.md
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/evaluation_metrics.json
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/evaluation_trace.png
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/evaluation_vs_training.png
```
