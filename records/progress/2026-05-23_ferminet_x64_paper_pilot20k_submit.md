# 2026-05-23 FermiNet X64 Paper-Geometry Pilot 20000-Step Submission

Created and submitted a FermiNet PBC C-diamond pilot to move closer to the
DeepSolid supplementary diamond benchmark while keeping the existing 10000-step
FP32 result intact.

Important caveat:

```text
True PBC-HF cc-pVDZ pretraining is not enabled in this FermiNet route yet.
Upstream FermiNet's pretraining path targets molecular PySCF orbitals, while
the current SolidNES FermiNet route uses PBC features, Ewald Hamiltonian, and
periodic multiwave envelopes. Submitting a nominal `pretrain_method: hf` job
here would not be a faithful periodic HF pretraining task.
```

Added files:

```text
src/solidnes/backends/ferminet_configs/diamond_pbc_gamma_paper.py
configs/system/diamond_c_primitive_gamma_paper.yaml
configs/sampler/metropolis_ferminet_pbc_mcmc20_burnin1000.yaml
configs/train/ground_state_ferminet_pbc_kfac_folx_batch4096_iter20000_no_pretrain.yaml
configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot.yaml
```

Also updated:

```text
scripts/backends/run_ferminet_train.py
```

The runner now reads the experiment runtime block before importing JAX and sets
`JAX_ENABLE_X64=1` when `runtime.x64_enabled: true`.

Configuration:

```text
experiment: diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot
geometry: DeepSolid supplementary paper geometry, a = 3.5738 Angstrom
primitive vector components: 1.7869 Angstrom
atoms: C at (0, 0, 0), C at (0.89345, 0.89345, 0.89345) Angstrom
network: FermiNet
hidden dims: (256, 32) x 4
determinants: 8
full determinant: true
complex output: false
optimizer: KFAC
laplacian: FOLX
Forward Laplacian enabled: true
batch size: 4096
iterations: 20000
MCMC burn-in: 1000
MCMC steps per iteration: 20
precision: x64 enabled
pretraining: none, pending true PBC-HF implementation
```

Build-only verification passed in the FermiNet venv:

```text
pretrain_method: None
pretrain_iterations: 0
mcmc_burn_in: 1000
target_jax_version: 0.10.1
precision_profile: fp64
x64_enabled: True
```

Scheduler compliance:

```text
submitter: scripts/slurm/submit_ferminet_gpu_smoke.sh
dry-run plan: tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/outputs/slurm_plans/ferminet_x64_burnin1000_iter20000_paper_pilot_g2_plan.json
blocked partitions: includes test
selected partition/node: intelgpu80g/gpu001
request: --gres=gpu:2 --cpus-per-task=96 --exclusive
time limit: 06:00:00
precision scheduling profile: fp64
```

Submitted job:

```text
job id: 127898
job name: solidnes-ferminet-x64-pilot20k
state after submission: RUNNING
node: gpu001
CUDA_VISIBLE_DEVICES: 0,1
JAX_ENABLE_X64: 1
jax: 0.10.1
jaxlib: 0.10.1
jax_default_backend: gpu
jax_devices: [CudaDevice(id=0), CudaDevice(id=1)]
start time: 2026-05-23 15:41:38 CST
```

Follow-up after completion:

```text
python scripts/validation/summarize_ferminet_benchmark.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot.yaml \
  --job-id 127898 \
  --log tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/logs/slurm/solidnes-ferminet-x64-pilot20k_127898.log \
  --err tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/logs/slurm/solidnes-ferminet-x64-pilot20k_127898.err \
  --plan-json tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/outputs/slurm_plans/ferminet_x64_burnin1000_iter20000_paper_pilot_g2_plan.json
```

Decision rule:

If the 20000-step x64 paper-geometry pilot lowers the tail estimate clearly
below the previous FP32 10000-step result `-75.2595080136 Ha`, continue toward
a 100000-step paper-reproduction run. If it stays near the same plateau, the
next engineering task is implementing a faithful PBC-HF cc-pVDZ pretraining
path rather than simply increasing the step count.

## Completed Result

Job `127898` completed successfully:

```text
state: COMPLETED
exit code: 0:0
elapsed: 02:23:08 by sacct
log runtime: 2026-05-23 15:41:38 to 18:04:37 CST, 8579 s
node: gpu001
resources: 2 GPU on intelgpu80g, 96 CPU
```

Generated outputs:

```text
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints/train_stats.csv
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints/qmcjax_ckpt_018349.npz
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/validation/benchmark_summary.json
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/validation/benchmark_summary.md
```

Main statistics:

```text
rows: 20000
first step: 0
last step: 19999
first energy: -16.6964320523 Ha
last energy: -75.4007786448 Ha
minimum energy: -75.5708224469 Ha at step 4349
last-10 mean: -75.4184587643 Ha
last-50 mean: -75.4161488268 Ha
last-500 mean: -75.4139125006 Ha
last-1000 mean: -75.4142544132 Ha
tail mean, last 10000 rows: -75.4114059555 Ha
tail block stderr, 5 blocks: 0.0010328659 Ha
mean pmove: 0.5446185901
tail pmove mean: 0.5354939661
last ewvar: 0.0001143532
seconds per optimization step including startup/burn-in/JIT: 0.428950
FOLX tile warnings: 0
tracebacks: 0
```

The five tail block means are:

```text
-75.4080928668
-75.4102141516
-75.4119173486
-75.4128095172
-75.4139958933 Ha
```

Comparison:

```text
DeepSolid supplementary diamond Net: -75.4009(2) Ha
FermiNet x64 pilot last-1000 mean:   -75.4142544132 Ha
difference:                         -0.0133544132 Ha

FermiNet x64 pilot tail mean:        -75.4114059555 Ha
difference from paper value:         -0.0105059555 Ha

Previous FP32 FermiNet 10k tail:     -75.2595080136 Ha
x64 pilot tail improvement:          -0.1518979419 Ha
```

Interpretation:

The x64 paper-geometry pilot decisively passed the decision rule. It reached
and slightly exceeded the DeepSolid supplementary diamond small-cell value by
training-tail estimates, despite not using true PBC-HF pretraining. This is
not yet a final publishable energy because it is not an independent fixed-
parameter inference run. The next accuracy step is to run a final evaluation
from the best checkpoint or the final checkpoint with a long inference chain,
then decide whether a longer 100000-step training run is still needed.
