# Progress

Last updated: 2026-06-01

## Current State

Project policy now requires every iterative training or evaluation task with
`iterations >= 1000` to save the final-step checkpoint. FermiNet/PsiFormer
runs launched through `run_ferminet_train.py` enforce this at runtime; DeepSolid
configs built through the adapter use step-based checkpointing for the same
rule unless explicitly overridden.

The DeepQMC-aligned excited-state method has been promoted into source-code
defaults. `--mainline-excited-state` now resolves to the 0103 fused-QKV
no-merge PsiFormer configuration with independent per-state parameters,
`merge_keys: []`, diagonal MCMC/local-energy/overlap-JVP paths, equal overlap
energy weighting, fixed state ordering, and KFAC state-count norm scaling
disabled. Non-empty `merge_keys` are still implemented and are classified as
`merge_key_variant`, so they remain available for explicit controls but are not
the production default. This milestone was committed and pushed as
`34d6574 Set no-merge excited-state mainline`.

Carbon-diamond benchmark reproduction is complete through both DeepSolid and
FermiNet. The FermiNet PBC-HF pretraining implementation and diamond-Gamma
validation milestone is also complete for the current cc-pVDZ workflow.

The project has moved from ground-state/pretraining route hardening into the
native FermiNet PBC excited-state method path. The Szabo and Noe JCTC 2024
overlap-penalty method is now represented in the native `vmc_overlap` path,
including optional spin penalty plumbing through FermiNet `spin_energy` and
`S^2` diagnostics. Spin-penalized runs now also write bare Hamiltonian energies
for paper-style excitation-gap reporting. The 1000-step and 2000-step spin
beta sweeps completed, but no tested beta is ready for a 10000-step production
run at the current optimizer settings. In the 2000-step sweep, `beta=0.008` is
the least bad continuation candidate; `beta=0.002` and `beta=0.018` are useful
controls.
Task `0087` completed as a single 2000-step pressure test with
`spin_penalty=10.0`. It suppresses the final spin diagnostic but destabilizes
the physical bare gap, so it is not a production candidate.
Task `0088`, the requested beta=0 100000-step baseline, also completed. It
provides the no-spin-penalty reference, but confirms that excited-state spin is
not controlled without an explicit fix: final `S^2` diagonal/trace is
`[1.4585, 81.4612] / 82.9197`, and the last 10000 finite frames include
139 frames with `|S^2 trace| > 10`.
Task `0089`, the fixed-ground follow-up requested after the 0088 ground-state
comparison, completed but was not production-ready. It trained one paper-size
x64 FermiNet state for 20000 beta=0/no-spin KFAC/FOLX iterations against the
fixed 0044 ground-state checkpoint. Job `129670` completed, but the final
physical energy fell below the fixed ground-state reference, giving an
unphysical negative gap.

Task `0094` is the new PsiFormer/self-attention implementation and benchmark
bundle under `tasks/psiformer/`. SolidNES now supports a configurable
PsiFormer attention implementation with `auto`, `ferminet`, and `fused_qkv`
options. The default PBC PsiFormer config uses `auto`, which now resolves
directly to a LapNet-style fused-QKV projection for GPU production runs. Local
CPU build-only and forward benchmarks passed; upstream and fused-QKV outputs
were identical on the test batch, but CPU timing is no longer used to choose
the default because subsequent PsiFormer calculations are GPU-only. The
corrected CUDA benchmark passed on the `test` GPU partition as
Slurm job `131644` after normalizing `gpu` to `cuda` for JAX platform
selection and fixing the scheduler planner so explicit `test` GPU submissions
can override the default blocked partition list. On 256 walkers,
fused-QKV matched upstream FermiNet outputs exactly and improved median forward
time from 0.000454 s to 0.000432 s, a 1.051x median speedup on `test001`/RTX
4090.

Task `0095`, the native PsiFormer training-path smoke and attention comparison,
also completed under `tasks/psiformer/`. Jobs `131661`, `131664`, `131666`,
and `131667` all completed on `test/test001`. The `auto` GPU smoke resolved
to `fused_qkv`; the matched batch512 full-training comparison was effectively
tied and slightly slower for fused-QKV (`36.855` vs `37.075` s/iter,
`0.994x`); and the fused-QKV batch1024 probe completed at `41.846` s/iter.
The conclusion is that the short native training loop is bottlenecked outside
the attention projection, while `auto -> fused_qkv` remains the production
default from the exact GPU forward benchmark.

Post-0095 policy update: because future PsiFormer calculations are GPU-only,
`model.attention.implementation: auto` now resolves directly to `fused_qkv`.
The explicit `ferminet` implementation remains available for controls, but CPU
timing no longer influences the default path.

Task `0096` is the active no-pretrain paper-scale PsiFormer attention
full-stack validation. It is allocated under
`tasks/psiformer/0096_psiformer_attention_full_stack/`, with paper-size
`auto`, explicit `ferminet`, and explicit `fused_qkv` configs prepared for
build-only checks, GPU forward exactness, and a short native KFAC smoke.
The full training-speed comparison completed as batch4096/iter10000 jobs with
Slurm `--partition amdgpu40g,amdgpu80g`. Jobs `131735` and `131736` both ran
on `amdgpu40g/gpu006`, each requested 4 GPUs and 64 CPU cores, and both exited
`0:0`; spin penalty and S2 observables were disabled. The upstream/FermiNet
attention job took 0.514879 s/iteration, while fused-QKV took 0.518262
s/iteration, so fused-QKV was about 0.657% slower end-to-end in the current
native KFAC/FOLX training path. The earlier fixed-partition submissions
`131692`--`131695` were redundant, and the first combined-partition 2000-step
submissions `131697`--`131698` were replaced by the 10000-step pair.
Per-step state-energy/gap diagnostics were added from `energy_matrix.npy`:
final gaps are 8.088 eV upstream and 4.685 eV fused-QKV, while last-1000 mean
gaps are 5.333 eV upstream and 7.932 eV fused-QKV.
Because those first 10000-step jobs used the speed precision profile, matching
fp64/no-TF32 reruns were added and submitted on 2026-05-28: job `131952`
upstream/FermiNet attention and job `131953` fused-QKV attention. The reruns use
`runtime.precision_profile=fp64`, `runtime.x64_enabled=true`,
`JAX_ENABLE_X64=1`, `psiformer.tf32=false`, the same batch4096/iter10000
training setup, and the same 4-GPU/64-CPU combined `amdgpu40g,amdgpu80g`
queue policy.
The first x64 attempt was replaced after upstream job `131952` exposed a FOLX
x64 sparse-mask warning at the PsiFormer spin-feature concatenate. SolidNES now
patches that input concatenate with a value-preserving zero-derivative spin
feature, and clean `x64_folxfix` jobs `131974` fused-QKV and `131975` upstream
were submitted with the same training and scheduler parameters.

## Active Step

See `ACTIVE_TASK.md` for the exact state and next command.

Short version:

- Active task: source-code mainline bookkeeping for the DeepQMC-aligned
  no-merge excited-state method.
- State: complete and pushed to `origin/main`.
- Latest result: `scripts/validation/check_excited_state_mainline_defaults.py`
  passed; build-only `--mainline-excited-state` resolves to the no-merge
  mainline; build-only merge-layers resolves to `merge_key_variant`; and
  `scripts/validation/check_ferminet_native_overlap_loss_alignment.py` passed.
- Evidence: GPU target/backend probes `0047--0050`, training integration and
  matched controls `0053--0062`.
- Completed in this step: cloned ignored `external/deepqmc/` at revision
  `f9e1ff5` and wrote
  `docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
- Completed in this step: implemented backend-independent overlap and penalty
  utilities under `src/solidnes/excited_states/` and verified them with
  `scripts/validation/check_excited_state_penalty_objective.py`.
- Completed in this step: implemented a minimal FermiNet PBC excited-state
  scaffold and verified it with
  `scripts/validation/check_ferminet_pbc_excited_scaffold.py`.
- Completed in this step: added and ran a build-only FermiNet/JAX adapter
  check at
  `scripts/validation/check_ferminet_pbc_excited_adapter_build.py`. It keeps
  `cfg.system.states == 0`, initializes two external state parameter trees,
  wraps `network.apply` into the scaffold interface, and constructs the PBC
  local-energy wrapper.
- Completed in this step: promoted the FermiNet/JAX adapter wrapper pattern
  into reusable source at
  `src/solidnes/excited_states/ferminet_pbc_adapter.py`; the validation script
  now calls this module.
- Completed in this step: connected the real FermiNet adapter to the penalty
  objective through `evaluate_ferminet_pbc_penalty_terms(...)` and verified it
  with `scripts/validation/check_ferminet_pbc_penalty_terms.py` using a cheap
  local-energy stand-in.
- Completed in this step: added a differentiable external-state penalty
  objective gradient and minimal SGD update scaffold, verified by
  `scripts/validation/check_ferminet_pbc_penalty_grad_step.py`.
- Completed in this step: added a build-only multi-step cheap-local-energy
  optimization smoke, verified by
  `scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py`.
- Completed in this step: scheduled and passed run `0063`, the first
  carbon-diamond Gamma two-state real PBC local-energy/Laplacian smoke.
  The full-node retry `128439` used `intelgpu80g/gpu001`, both A100 80GB GPUs,
  96 CPU cores, and an exclusive allocation; it completed with exit code `0:0`.
- Completed in this step: added reusable fixed-sample two-state FermiNet PBC
  penalty-objective training-loop helpers under
  `src/solidnes/excited_states/ferminet_pbc_training.py`; the multi-step cheap
  optimization smoke now exercises this source path.
- Attempted in this step: run `0064`, the first real-local-energy two-step
  fixed-sample training-loop smoke. Job `128523` started on
  `intelgpu80g/gpu001` with both A100 80GB GPUs and 96 CPU cores, then failed
  after `00:04:57` because final validation found
  `final_local_energy = [[nan], [nan]]`.
- Completed after 0064 failure: replaced the direct real-local-energy
  `value_and_grad` SGD path with a paper-tangent update path: lower-state
  stop-gradient overlap behavior, psi-ratio clipping, local-energy clipping,
  automatic overlap-gradient scaling from gap/std diagnostics, optional global
  gradient clipping, and finite-gradient/update/candidate-term guards before
  committing parameters.
- Validation after source fix: py_compile, no-compute penalty objective check,
  no-compute FermiNet scaffold check, FermiNet/JAX adapter build check, penalty
  terms check, gradient-step check, and multi-step cheap-local-energy
  optimization smoke all passed. Record:
  `records/progress/2026-05-24_ferminet_pbc_paper_tangent_guards.md`.
- Completed in this step: diagnosed run `0065` failure
  `penalty_objective_step_0 = nan`, fixed the paper-tangent objective to reuse
  precomputed penalty terms outside `value_and_grad`, added forward-safe
  surrogate diagnostics, and added a cheap-local-energy regression proving no
  extra local-energy calls happen inside `value_and_grad`.
- Completed in this step: diagnosed run `0066` zero-gradient failure as a
  degenerate one-walker smoke for seed `47`; local cheap smoke reproduced the
  issue with `walkers=1` and passed with `walkers=2`.
- Completed in this step: scheduled and passed run `0067`, the two-walker
  fixed-sample real PBC local-energy/Laplacian paper-tangent training smoke.
  Job `128677` completed on `intelgpu80g/gpu001` with both A100 80GB GPUs, 96
  CPU cores, and exit `0:0`. Both guarded updates were accepted with finite
  gradient norms `557.018` and `555.336`.
- Completed in this step: added the sampler-integrated FermiNet PBC
  excited-state driver and passed scheduled real-local-energy driver smokes.
  Run `0068` job `128751` passed with two walkers per state in `00:05:44`.
  Run `0069` job `128752` passed with four walkers per state in `00:04:31`.
  Both used `intelgpu80g/gpu001`, both A100 80GB GPUs, 96 CPU cores, real PBC
  local energy, accepted guarded updates, finite sampler diagnostics, and
  successful checkpoint roundtrips.
- Completed in this step: added the production backend runner and passed run
  `0070`, the first 12-iteration controlled real-local-energy driver
  trajectory with four walkers per state. Job `128758` completed in
  `00:14:28`, wrote checkpoints at iterations 4, 8, and 12, accepted every
  guarded update, and ended with objective `-13.346862`, final state energies
  `[-17.577646, -11.450842]`, and final off-diagonal overlap `0.483194`.
- Completed in this step: added the post-0070 optimizer/sharing/numerical/driver
  source iteration. The external-state path now supports SGD, Adam, LAMB, and
  direct KFAC through `kfac_jax.Optimizer`; carries EWMA state-energy/std
  running stats for overlap scaling; projects selected shared parameter leaves;
  separates optimizer-update and sharing-projection diagnostics; checkpoints
  optimizer state and running stats; and supports candidate-check cadence plus
  update caps in the driver runner and validation scripts. CPU cheap-local-energy
  smokes passed for Adam, LAMB, KFAC, sampler-integrated KFAC checkpoint
  roundtrip, and KFAC runner resume. Record:
  `records/progress/2026-05-25_ferminet_pbc_optimizer_sharing_numeric_driver_update.md`.
- Completed in this step: run `0071`, the scheduled direct-KFAC real-local-energy
  FermiNet PBC driver smoke. The config is
  `configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_smoke.yaml`;
  it uses two external states, two walkers per state, two driver iterations,
  paper-tangent objective terms, EWMA overlap scaling, shared `layers/streams`
  parameters, KFAC damping/norm constraint `1e-3`, candidate checks every
  iteration, and update cap `1e-3`. A CPU cheap-local-energy config check
  passed. The SLURM dry-run selected `amdgpu40g/gpu006` with one A100 40GB GPU
  and 8 CPU cores; job `129088` completed in `00:07:51` with exit `0:0`.
  The driver completed 2/2 iterations, wrote checkpoints at iterations 1 and
  2, accepted both KFAC updates, kept optimizer update norms at about `1e-3`,
  and ended with objective `-7.182288`, final state energies
  `[-7.621417, -8.340336]`, and final off-diagonal overlap `-0.399647`.
- Run `0072`, the first 100-iteration direct-KFAC real-local-energy full-node
  trial, was cancelled by request after `00:12:02` because it used the
  pre-multi-device KFAC bridge. JAX saw all four A100 40GB GPUs on
  `amdgpu40g/gpu005`, but the KFAC optimizer was still configured with
  `multi_device=False`; checkpoint iteration 10 was written, but no final
  summary was produced.
- Run `0073`, the multi-GPU direct-KFAC external-wrapper trial, reached
  checkpoints 10/20/30 with KFAC metadata `multi_device=true`, but sampled GPU
  utilization stayed near zero while the Python process was CPU-bound. It was
  cancelled after `00:23:45` to stop optimizing the external wrapper.
- Run `0074` moved the method into FermiNet's native excited-state
  architecture: `cfg.system.states=2`, `cfg.optim.objective=vmc_overlap`, and
  native KFAC `multi_device=True`. The first job exposed the upstream FermiNet
  PBC Hamiltonian gap (`Excited states not implemented with PBC`), so SolidNES
  now supplies the missing PBC excited-state local-energy branch in
  `src/solidnes/backends/ferminet_pbc_hamiltonian.py`. Retry job `129219`
  completed 20 native KFAC steps on `amdgpu40g/gpu005` in `00:01:55`, with
  final energy `-22.488228`, EW mean `-22.557020`, mean pmove `0.910938`, and
  final state-energy vector `[-22.203577, -23.186222]`.
- Completed in this step: run `0075`, the native `vmc_overlap` KFAC smoke with
  batch1024 after noting that smaller sample counts can hide GPU utilization.
  Job `129240` completed on `amdgpu40g/gpu005` in `00:01:59` with exit `0:0`.
  Native KFAC registered loss shape `float32[256,2]` per device over four
  GPUs; the run wrote `train_stats.csv`, `energy_matrix.npy`, and
  `overlap_matrix.npy`. Final energy was `-22.453096`, final EW mean was
  `-22.400412`, mean pmove was `0.911453`, final state-energy vector was
  `[-22.497761, -22.392517]`, and final overlap matrix was
  `[[1.0, 0.0773164], [0.123951, 1.0]]`.
- Completed in this step: run `0076`, the native FermiNet PBC excited-state
  `vmc_overlap` KFAC speed/stability baseline with batch4096 and 50
  iterations. The build-only adapter check passed with `states=2`,
  `objective=vmc_overlap`, `batch_size=4096`, and `iterations=50`. Job
  `129249` completed on `amdgpu40g/gpu005` in `00:03:14` with exit `0:0`.
  JAX saw all four A100 40GB GPUs; KFAC registered per-device loss shape
  `float32[1024,2]`; 50 rows were written. Final energy was `-24.691084`,
  final EW mean was `-24.117025`, mean pmove was `0.910310`, final
  state-energy vector was `[-25.515295, -23.108868]`, and final overlap matrix
  was `[[1.0, 0.0672966], [0.328056, 1.0]]`. The backend window was 149s
  (`2.98 s/step`), but this short-run speed still includes startup, burn-in,
  and compilation overhead. Record:
  `records/progress/2026-05-25_ferminet_native_pbc_excited_state_batch4096_iter50.md`.
- Completed in this step: the native FermiNet overlap loss was aligned with the
  Szabo-Noe/DeepQMC settings: `alpha=4.0`, `scale_by=max_gap_std`,
  psi-ratio clipping, median local-energy clipping, energy-based lower-state
  ordering, and overlap scale/order diagnostics. CPU helper validation,
  build-only config checks, and `git diff --check` passed. Run `0077` then
  validated the paper-aligned path on `amdgpu40g/gpu005` with four A100 40GB
  GPUs, batch4096, native KFAC, and 5 iterations. Job `129257` completed in
  `00:02:04` with exit `0:0`; final energy was `-22.351885`, final
  state-energy vector was `[-22.548399, -21.983301]`, final overlap matrix was
  `[[1.0, 0.0314396], [0.0647455, 1.0]]`, final overlap-gradient scale was
  `[[5.0, 5.0], [5.0, 5.0]]`, and final state ordering was `[0, 1]`. Record:
  `records/progress/2026-05-25_ferminet_native_overlap_loss_paper_alignment.md`.
- Completed in this step: run `0078`, the longer paper-aligned native FermiNet
  PBC excited-state trajectory with batch4096 and 1000 iterations. The
  build-only check passed; the Slurm dry-run was forced to `amdgpu40g` by
  blocking `amdgpu80g`; job `129262` completed on `amdgpu40g/gpu005` with
  four A100 40GB GPUs, 64 CPU cores, exclusive allocation, and exit `0:0`.
  It wrote 1000 rows. Final energy was `-73.959910`, final EW mean was
  `-73.935610`, final EW variance was `0.00348556`, tail-100 mean energy was
  `-73.867186`, final state-energy vector was `[-74.176765, -73.526321]`,
  final overlap matrix was `[[1.0, 0.0186032], [-0.0343843, 1.0]]`, final
  overlap-gradient scale was `[[5.0, 5.0], [5.0, 5.0]]`, and final state
  ordering was `[0, 1]`.
  Record:
  `records/progress/2026-05-25_ferminet_native_overlap_paper_aligned_1000_submit.md`.
  Completion record:
  `records/progress/2026-05-25_ferminet_native_overlap_paper_aligned_1000_complete.md`.
- Completed in this step: method/implementation cleanup for the native
  FermiNet PBC `vmc_overlap` path, excluding spin penalty. The SolidNES adapter
  now has a `szabo_noe_2024_penalty` method profile; native FermiNet loss data
  carries fixed-shape EWM energy/std scale inputs; KFAC norm constraints are
  scaled by state count; raw, symmetrized, and squared-penalty overlap
  diagnostics are logged separately; and the summary tool reports the new
  diagnostics. The first 10000-step submit, job `129267`, failed after step 0
  because KFAC saw different traces when EWM fields changed from `None` to
  arrays. Fixed-shape NaN EWM arrays were then initialized before KFAC
  construction. Retry run `0079`, job `129272`, completed on
  `amdgpu40g/gpu005` in `00:20:11` with exit `0:0`, wrote 10000 rows, and
  ended with final energy `-74.583840`, EW mean `-74.684850`, EW variance
  `0.00702351`, mean pmove `0.549417`, final state energies
  `[-74.792580, -74.169594]`, final symmetric overlap matrix
  `[[1.0, 0.0159107], [0.0159107, 1.0]]`, and final overlap penalty matrix
  `[[1.0, 0.000253152], [0.000253152, 1.0]]`.
  Record:
  `records/progress/2026-05-25_ferminet_native_method_cleanup_and_10000.md`.

## Completed Structural Cleanup

- Legacy generated artifacts were migrated into numbered `tasks/` bundles.
- The retired target-specific scaffold was removed.
- `tasks/TASK_LEDGER.md` is the readable task ledger.
- `records/run_index.md` is the run-number allocator; the next available run
  number is `0080`.
- Top-level `results/`, `outputs/`, and `logs/` are retired and should not be
  recreated for new work.

## Evidence Snapshot

DeepSolid direct route:

- Job `127816`
- Tail-2000 mean: `-75.4161279970 Ha`
- Paper reference: `-75.4009 Ha`

FermiNet route:

- Training job `127898`
- Evaluation job `127992`
- Fixed-parameter evaluation mean: `-75.4125655570 Ha`
- Paper reference: `-75.4009 Ha`

FermiNet PBC-HF pretraining:

- Implementation record:
  `records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md`
- Milestone record:
  `records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md`
- JAX PBC GTO cc-pVDZ validation at image cutoff `3`: AO max abs `1.12e-9`,
  occupied-MO max abs `8.51e-10`.
- JAX PBC GTO cc-pVDZ pretraining target was about `2.52x` faster overall than
  the PySCF target benchmark for the current probe.
- Fixed-iteration early training favored pretraining; short wall-clock
  timeboxed comparisons were mixed, so pretraining is a validated option rather
  than a universal default.

Next-phase task area:

- `tasks/excited_state_nesvmc/`
- Purpose: future numbered task bundles for reproducing the paper-style
  penalty-based excited-state VMC method and testing it on specific periodic
  materials.

Reference audit:

- `docs/05_reference_projects/deepqmc_penalty_excited_states.md`
- Local ignored checkout: `external/deepqmc/`
- Inspected DeepQMC revision: `f9e1ff5`

Current implementation scaffold:

- `src/solidnes/excited_states/overlap.py`
- `src/solidnes/excited_states/penalty.py`
- `src/solidnes/excited_states/ferminet_pbc_adapter.py`
- `src/solidnes/excited_states/ferminet_pbc_scaffold.py`
- `src/solidnes/excited_states/ferminet_pbc_training.py`
- `src/solidnes/excited_states/ferminet_pbc_driver.py`
- `scripts/validation/check_excited_state_penalty_objective.py`
- `scripts/validation/check_ferminet_pbc_excited_scaffold.py`
- `scripts/validation/check_ferminet_pbc_excited_adapter_build.py`
- `scripts/validation/check_ferminet_pbc_penalty_terms.py`
- `scripts/validation/check_ferminet_pbc_penalty_grad_step.py`
- `scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py`
- `scripts/validation/check_ferminet_pbc_real_local_energy_smoke.py`
- `scripts/validation/check_ferminet_pbc_driver_smoke.py`
- `scripts/validation/summarize_ferminet_pbc_driver_run.py`
- `scripts/validation/summarize_ferminet_native_excited_run.py`
- `scripts/validation/check_ferminet_native_overlap_loss_alignment.py`
- `scripts/backends/run_ferminet_pbc_excited_driver.py`
- `records/progress/2026-05-24_excited_state_penalty_utilities.md`
- `records/progress/2026-05-24_ferminet_pbc_excited_scaffold.md`
- `records/progress/2026-05-24_ferminet_pbc_excited_adapter_build_check.md`
- `records/progress/2026-05-24_ferminet_pbc_excited_adapter_source.md`
- `records/progress/2026-05-24_ferminet_pbc_penalty_terms.md`
- `records/progress/2026-05-24_ferminet_pbc_penalty_grad_step.md`
- `records/progress/2026-05-24_ferminet_pbc_penalty_opt_smoke.md`
- `records/progress/2026-05-24_ferminet_pbc_real_local_energy_smoke.md`
- `records/progress/2026-05-24_ferminet_pbc_training_loop.md`
- `records/progress/2026-05-25_ferminet_pbc_driver_smokes.md`
- `records/progress/2026-05-25_ferminet_pbc_driver_controlled12.md`
- `records/progress/2026-05-25_ferminet_pbc_optimizer_sharing_numeric_driver_update.md`
- `records/progress/2026-05-25_ferminet_native_pbc_excited_state_smoke.md`
- `records/progress/2026-05-25_ferminet_native_pbc_excited_state_batch4096_iter50.md`
- `records/progress/2026-05-25_ferminet_native_overlap_loss_paper_alignment.md`
- `records/progress/2026-05-25_ferminet_native_overlap_paper_aligned_1000_submit.md`
- `records/progress/2026-05-25_ferminet_native_overlap_paper_aligned_1000_complete.md`
- `records/progress/2026-05-25_ferminet_native_method_cleanup_and_10000.md`
- `records/progress/2026-05-25_ferminet_native_deepqmc_ewm_alignment.md`

## Current Rules

- Compute and validation artifacts go under one numbered task bundle:
  `tasks/<phase>/<system>/<setup_or_kpoint>/<task_type>/NNNN_short_slug/`.
- Pure source audits, literature review, external-code inspection, and design
  notes do not consume a run number and should go under `docs/` or
  `records/progress/`.
- Each task bundle contains `results/`, `outputs/`, and `logs/`.
- Matched sweeps/ablations should use one numbered task bundle with
  `runs/<variant>/` subdirectories and aggregate comparison artifacts in the
  parent `results/validation/`.
- SLURM submitters require `SOLIDNES_TASK_ROOT` or explicit plan/log paths.
- Every completed or materially updated task must update
  `tasks/TASK_LEDGER.md`.
- Substantial completed work should also write a dated note under
  `records/progress/`.

## History

Detailed dated history lives in `records/progress/`.

The previous long top-level progress file was archived at:

```text
records/progress/2026-05-23_progress_snapshot_before_top_level_slimming.md
```
