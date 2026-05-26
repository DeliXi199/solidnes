# 2026-05-26 FermiNet fixed-ground symmetric-overlap beta=0 submission

## Scope

- Task: `0090_ferminet_fixed_ground_symmetric_overlap_beta0_iter20000`
- Experiment config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_fixed_ground_symmetric_overlap_beta0_iter20000.yaml`
- Train config:
  `configs/train/excited_state_ferminet_pbc_fixed_ground_symmetric_overlap_beta0_iter20000.yaml`
- Objective: `fixed_ground_overlap`
- Symmetric fixed-ground sampling: enabled
- Iterations: `20000`
- Spin penalty: `0.0`
- `S^2` observable: disabled

## Code changes

- Added independent fixed-ground walker fields to FermiNet training data.
- Added symmetric fixed-reference overlap diagnostics and loss support.
- Added a fixed-ground MCMC update path that advances the trainable excited
  walkers under trainable parameters and the fixed ground walkers under the
  checkpoint parameters.
- Ensured fixed-reference walker buffers are copied before entering donated
  pmapped steps, avoiding JAX donated-buffer aliasing between trainable and
  fixed-reference fields.
- Passed `fixed_ground.symmetric_sampling` through the SolidNES FermiNet
  adapter and summaries.

## Validation

- `py_compile` passed for the changed FermiNet and adapter modules.
- `build_ferminet_config.py --create-output-dirs` passed.
- `run_ferminet_train.py --build-only` passed.
- A local CPU `jax.value_and_grad` dummy check of the symmetric fixed-ground
  overlap loss returned finite value, diagnostics, and gradients.
- A two-CPU-device `pmap(..., donate_argnums=(0,))` data donation check passed
  after the fixed-reference buffer-copy fix.

## Slurm submission

- Requested exact partition: `gpu80gllm`
- Direct `gpu80gllm` attempt: job `130202`, cancelled after Slurm kept it in
  `PD (PartitionConfig)` because `gpu80gllm` does not allow the current
  account.
- Intermediate submitted job: `130203` on `amdgpu80g`, cancelled while pending
  after the user requested `amdgpu40g`.
- First `amdgpu40g` submitted job: `130207`, failed in burn-in with JAX
  donated-buffer aliasing before the buffer-copy fix.
- Effective submitted job: `130209`
- Effective partition/node: `amdgpu40g/gpu007`
- Resources: one exclusive node, `gpu:4`, `64` CPU cores, `--mem=0`,
  walltime `12:00:00`.
- State after submission: `RUNNING`.
