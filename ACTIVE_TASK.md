# SolidNES Active Task

Last updated: 2026-05-28, Asia/Shanghai

## Purpose

This file records the exact small step currently in progress.

Use it as the handoff point when asking:

```text
Now what is the task doing?
What is the next concrete action?
Is a Slurm job running?
What result are we waiting for?
What condition marks this step complete?
```

Long-term milestones belong in `CURRENT_STATUS.md`. Chronological completed
work belongs in `records/progress/`.

## Current Small Step

```text
Step name: PsiFormer attention full-stack validation without pretraining
State: x64 FOLX-fix rerun submitted
Current action: the same two full-node no-pretrain PsiFormer attention comparisons were resubmitted with `runtime.precision_profile=fp64`, `runtime.x64_enabled=true`, `JAX_ENABLE_X64=1`, and `psiformer.tf32=false`. The first x64 attempt (`131952`/`131953`) was cancelled after `131952` exposed a FOLX x64 sparse-mask warning at the PsiFormer spin-feature concatenate. SolidNES now patches that concatenate with a value-preserving zero-derivative spin feature. Clean replacement jobs are `131974` fused-QKV attention running on `amdgpu40g/gpu006` and `131975` upstream/FermiNet attention pending in the combined `amdgpu40g,amdgpu80g` queue. Both request 4 GPUs, 64 CPU cores, batch4096, 10000 iterations, KFAC, FOLX, no spin penalty, and no S2 observables.
Current blocker: waiting for the two fp64 FOLX-fix reruns to finish before regenerating the speed and gap comparison plots.
Backend: FermiNet
System: carbon diamond primitive cell first, then selected material tests
Goal: reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC
      method in the SolidNES code path, then test it on concrete periodic
      materials.
```

## Current Position

```text
Benchmark reproduction is complete for both DeepSolid and FermiNet.
Task 0096 is now active for the no-pretrain paper-scale PsiFormer attention
full-stack validation. It lives under
`tasks/psiformer/0096_psiformer_attention_full_stack/`. Test-partition flow
checks passed: GPU forward exactness job `131685` completed, and native KFAC
smoke job `131686` completed with `auto -> fused_qkv`. Full-node 10000-step
speed jobs `131735` and `131736` completed on `amdgpu40g/gpu006` with exit
`0:0`; fused-QKV was about 0.657% slower end-to-end than upstream attention.
Those first 10000-step jobs used the speed precision profile, so fp64/no-TF32
reruns were submitted as jobs `131952` upstream attention and `131953`
fused-QKV attention with the same training size and scheduler request.
Job `131952` exposed a FOLX x64 sparse-mask warning in the PsiFormer spin-feature
concatenate, so that runtime patch was fixed and clean `x64_folxfix` jobs
`131974` fused-QKV and `131975` upstream were submitted.
Per-step state-gap plots were added from `energy_matrix.npy`: the final
single-step gap is smaller for fused-QKV, but the last-1000 mean gap is larger
for fused-QKV, so tail/rolling statistics are more informative than the final
row alone.
Task 0095 completed the PsiFormer native training-path validation step. It
lives under `tasks/psiformer/0095_psiformer_native_training_smoke/` and holds
variant subruns for `auto_smoke`, `ferminet_b512`, `fused_qkv_b512`, and
`fused_qkv_b1024`. Jobs `131661`, `131664`, `131666`, and `131667` completed
on `test/test001`. The GPU `auto` policy resolved to `fused_qkv`; b512
full-training timing was effectively tied and slightly slower for fused-QKV
(`36.855` vs `37.075` s/iter, `0.994x`), so short native training is dominated
outside the attention projection. The next available run number is 0097.
Task 0094 is the completed PsiFormer/self-attention implementation step. It lives
under `tasks/psiformer/0094_psiformer_attention_build_benchmark/`, per the
requested task organization. Build-only config passed with
`network_type=psiformer`, `objective=vmc_overlap`, `states=2`, and
`psiformer_attention_implementation=auto`. Local CPU forward benchmarking
confirmed exact output agreement between upstream FermiNet attention and the
LapNet-style fused-QKV implementation on the tested batch. The default `auto`
policy now selects fused-QKV directly because future PsiFormer calculations are
GPU-only; the upstream FermiNet implementation remains available only through
the explicit `ferminet` control setting. The first GPU benchmark job
`131634` failed before model execution due to an invalid JAX platform flag
(`gpu` instead of `cuda`). After platform normalization and a scheduler planner
fix allowing explicit `test` GPU submissions, job `131644` ran on
`test/test001` with one RTX 4090, 8 CPU cores, and 15 minutes. It completed in
about 24 seconds. For 256 walkers, upstream median forward time was
0.000454 s and fused-QKV median was 0.000432 s, a 1.051x median speedup with
`max_abs_logabs_delta=0.0` and `max_abs_sign_delta=0.0`.
Task 0086 completed the 12 beta values
`0.000,0.001,0.002,0.005,0.008,0.010,0.012,0.015,0.018,0.020,0.025,0.030`
as full-node `amdgpu80g` jobs, batch4096, 2000 iterations. Jobs 129314 and
129327--129337 all completed with exit code `0:0` on `gpu002`.
No tested low beta is production-ready at the current KFAC settings. `beta=0.008`
is the least bad continuation candidate, while `beta=0.002` and `beta=0.018`
are the most useful low/high controls; `beta=0.012` and `beta=0.030` had
transient non-finite `S^2`/bare-energy diagnostic frames.
Task 0087 completed the deliberately large `spin_penalty=10.0` 2000-step
pressure test as Slurm job `129431` on `amdgpu80g/gpu002` with exit `0:0` in
`00:07:46`. It wrote all 2000 rows and no non-finite arrays, but exposed a
physical bare-gap instability: final bare gap 8.458 eV, final training gap
2.793 eV, tail200 bare-gap median 4.520 eV, and tail200 mean/std
-119.552 +/- 1368.586 eV because step 1946 produced a -18445.091 eV bare-gap
spike. Final `S^2` diagonal [0.0109, -0.0099] shows the large spin penalty
suppresses the final spin diagnostic, but not enough to stabilize the physical
excitation gap.
FermiNet PBC-HF pretraining is implemented, GPU-tested, and validated for the
current carbon-diamond Gamma cc-pVDZ workflow.
JAX PBC GTO target evaluation is validated for the current diamond Gamma
cc-pVDZ workflow with image cutoff 3.
Fixed-iteration 1000-step comparisons favor pretraining, but matched
short-wall-clock comparisons are mixed.
Global agent instructions live in AGENTS.md and must be read at the start of
every new answer or work session.
Numbered task bundle 0082 contains the earlier 1000-step spin beta sweep
variants under `runs/<variant>/`. Labels 0083--0085 were consumed as Slurm job
names before the consolidation and should not be reused. Numbered task bundle
0086 contains the completed 2000-step, 12-point beta sweep. 0087 contains the
completed beta=10 pressure test. 0088 contains the completed and analyzed
beta=0 100000-step baseline. Tasks 0089--0093 contain the fixed-ground and
native-route follow-ups. Task 0094 is the completed PsiFormer/self-attention
bundle. Task 0095 is the completed PsiFormer native training smoke/comparison
bundle. Task 0096 is the active no-pretrain PsiFormer attention full-stack
bundle with the fp64 FOLX-fix rerun now queued/running; the next available run
number is 0097.
Native FermiNet spin penalty is now wired through SolidNES configs:
`spin_penalty` maps to `cfg.optim.spin_energy`, `observables_s2` writes
`s2_matrix.npy`, and spin-penalized runs write `bare_energy_matrix.npy` for
paper-style physical energy-gap reporting.
The first 1000-step spin beta sweep found no production-ready value at the
current learning-rate/KFAC settings: `beta=0.02` was finite but noisy with an
unstable final gap sign, `beta=0.05` produced NaNs after about step 846, and
`beta=0.10/0.20` produced finite but much too large bare gaps.
The 2000-step follow-up sweep and the beta=10 pressure test found no
production-ready spin-penalty setting at the current optimizer settings.
Task 0088 is the requested long beta=0 baseline: 100000 native FermiNet PBC
excited-state `vmc_overlap` iterations, batch4096, KFAC, overlap alpha 4.0,
`max_gap_std` scaling, `spin_penalty=0.0`, and `S^2` diagnostics enabled.
Slurm job `129450` completed on `amdgpu80g/gpu002` with exit `0:0` in
`03:46:24`. It wrote all 100000 rows. Final state energies are
`[-75.096855, -74.919098] Ha`, final gap is `4.837 eV`, and tail200 gap
median is `9.205 eV`. The spin diagnostic is not controlled: final `S^2`
diagonal/trace is `[1.4585, 81.4612] / 82.9197`, full-run `S^2` diagnostics
contain 35 non-finite frames, and the last 10000 finite frames include
139 frames with `|S^2 trace| > 10`. Analysis and plots are under
`tasks/excited_state_nesvmc/0088_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta0000_iter100000/results/validation/`.
Task 0089 was the fixed-ground follow-up requested after the 0088 ground-state
comparison. It completed as job `129670`, but the route is not production-ready
because the final physical energy was below the fixed ground-state reference,
giving an unphysical negative excitation gap.
The direct real-local-energy `value_and_grad` SGD path has been replaced with a
paper-tangent penalty-VMC update path for the fixed-sample helper.
DeepQMC was cloned for source inspection under ignored `external/deepqmc/`.
The reference-source audit is recorded in
`docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
Backend-independent overlap/penalty utilities are implemented under
`src/solidnes/excited_states/`, with a no-compute synthetic validation script at
`scripts/validation/check_excited_state_penalty_objective.py`.
A minimal FermiNet PBC excited-state scaffold is implemented at
`src/solidnes/excited_states/ferminet_pbc_scaffold.py`, with a no-compute
synthetic validation script at
`scripts/validation/check_ferminet_pbc_excited_scaffold.py`.
The first real FermiNet/JAX build-only adapter check is implemented at
`scripts/validation/check_ferminet_pbc_excited_adapter_build.py` and passed in
the `solidnes-ferminet-jax0101-cuda12` environment. It keeps
`cfg.system.states == 0`, initializes two external state parameter trees, wraps
`network.apply` into the scaffold wavefunction-matrix interface, and constructs
the PBC local-energy wrapper without evaluating the expensive Laplacian path by
default.
The FermiNet/JAX wrapper pattern has been promoted from the validation script
into reusable source at
`src/solidnes/excited_states/ferminet_pbc_adapter.py`. The validation script now
calls this module instead of carrying a duplicate implementation.
The real FermiNet adapter is now connected to the penalty objective through
`evaluate_ferminet_pbc_penalty_terms(...)`, which returns state energies,
wavefunction ratios, overlap diagnostics, and penalty terms. The build-only
validation script
`scripts/validation/check_ferminet_pbc_penalty_terms.py` passed with a cheap
local-energy stand-in.
The first differentiable optimization-step scaffold is implemented through
`value_and_grad_ferminet_pbc_penalty_objective(...)` and
`apply_external_state_sgd_step(...)`. The build-only validation script
`scripts/validation/check_ferminet_pbc_penalty_grad_step.py` passed with a
cheap local-energy stand-in and confirmed finite nonzero gradients plus a real
parameter update.
The first build-only multi-step optimization smoke is implemented at
`scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py`. It uses the real
FermiNet PBC external-state adapter with a cheap local-energy stand-in and
confirmed three consecutive finite SGD updates with decreasing penalty
objective, without creating a numbered task bundle.
Future small task bundles for this phase should go under
tasks/excited_state_nesvmc/ when they produce build, smoke, experiment,
evaluation, analysis, SLURM, log, or result artifacts. Pure reference-source
audits and design notes do not consume a run number.
Run 0063 is allocated for the first scheduled real PBC local-energy/Laplacian
smoke:
`tasks/excited_state_nesvmc/0063_ferminet_pbc_real_local_energy_smoke/`.
The dry-run plan passed and selected one A100 80GB GPU on `intelgpu80g`,
node `gpu001`, with 8 CPU cores and a 20-minute walltime.
Run 0063 was first submitted as Slurm job `128435` with 1 GPU and 8 CPU
cores; it completed successfully. Per user request, the task was resubmitted
as full-node Slurm job `128439` on `intelgpu80g/gpu001`, using both A100 80GB
GPUs, 96 CPU cores, and an exclusive allocation.
The full-node real PBC local-energy/Laplacian smoke passed. Job `128439`
completed in 00:02:05 with exit code `0:0`; JAX reported `cuda:0` and
`cuda:1`; the validation summary recorded finite `[2, 1]` local energies,
finite `[2]` state energies, a finite `2x2` overlap matrix, and a finite
scalar penalty objective.
The first reusable fixed-sample training-loop helper is implemented at
`src/solidnes/excited_states/ferminet_pbc_training.py`. It owns external
state-parameter SGD updates, returns per-step penalty/energy/overlap/collapse
diagnostics, and is verified through
`scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py` with a cheap
local-energy stand-in.
After run 0064 failed, the training path was updated with ordered lower-state
stop-gradient overlap behavior, psi-ratio clipping, local-energy clipping,
automatic overlap-gradient scaling from gap/std diagnostics, optional global
gradient clipping, finite-gradient/update guards, and candidate-term finite
checks before parameter commits. The no-compute scaffold check, FermiNet/JAX
adapter build check, penalty terms check, gradient-step check, and multi-step
cheap-local-energy optimization smoke all pass after this source change. The
source-update record is
`records/progress/2026-05-24_ferminet_pbc_paper_tangent_guards.md`.
Run 0064 was allocated for the first scheduled real-local-energy multi-step
training-loop smoke:
`tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke/`.
The run used
`scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py`
with two external state parameter trees, one walker per state, and two SGD
steps through the real PBC local-energy/Laplacian path.
The dry-run plan passed. Because `intelgpu80g/gpu001` was busy, the planner
generated a queued full-node plan on `intelgpu80g` with `gpu:2`, 96 CPU cores,
exclusive allocation, and a 30-minute walltime.
Run 0064 was submitted as Slurm job `128523`. It started on
`intelgpu80g/gpu001`, ran for `00:04:57`, and failed with exit code `1:0`.
JAX saw both A100 80GB GPUs, but final validation raised
`ValueError: final_local_energy contains non-finite values: [[nan] [nan]]`.
The failure summary is recorded in
`tasks/excited_state_nesvmc/0064_ferminet_pbc_real_local_energy_training_smoke/results/validation/real_local_energy_training_smoke_failure.md`.
Run 0065 was submitted as Slurm job `128674`. It started on
`intelgpu80g/gpu001`, ran for `00:06:15`, and failed with exit code `1:0`.
JAX saw both A100 80GB GPUs, but validation raised
`ValueError: penalty_objective_step_0 is not finite: nan`. The failure summary
is recorded in
`tasks/excited_state_nesvmc/0065_ferminet_pbc_paper_tangent_training_smoke/results/validation/real_local_energy_training_smoke_failure.md`.
After the 0065 failure, the paper-tangent path was updated so the gradient
objective reuses precomputed true penalty terms outside `value_and_grad`, avoids
calling the real local-energy/Laplacian under the parameter-gradient transform,
records the true pre-update penalty objective separately from the gradient
objective, and uses a forward-safe zero-primal surrogate correction. Local
checks pass, including a regression that confirms no extra local-energy calls
inside `value_and_grad`.
Run 0066 has been allocated for the fixed scheduled real-local-energy
multi-step smoke:
`tasks/excited_state_nesvmc/0066_ferminet_pbc_paper_tangent_training_smoke_fix/`.
The approved FermiNet GPU submitter dry-run passed and selected
`intelgpu80g/gpu001`, `gpu:2`, 96 CPU cores, an exclusive allocation, and a
30-minute walltime. The job was submitted as Slurm job `128675`.
Run 0066 failed after `00:05:01` with
`ValueError: grad_l2_norm_step_0 must be positive, got 0.0`. The 0065 NaN
objective issue was fixed; local checks reproduced the zero-gradient condition
with `seed=47, walkers=1` and passed with `walkers=2`.
Run 0067 has been allocated for the same fixed real-local-energy multi-step
smoke with two walkers per state:
`tasks/excited_state_nesvmc/0067_ferminet_pbc_paper_tangent_training_smoke_walkers2/`.
The approved FermiNet GPU submitter dry-run passed and selected
`intelgpu80g/gpu001`, `gpu:2`, 96 CPU cores, an exclusive allocation, and a
30-minute walltime. The job was submitted as Slurm job `128677`.
Run 0067 completed successfully in `00:05:04` with exit code `0:0`. The
validation summary recorded finite `[2, 2]` real local-energy matrices, finite
state energies, finite overlap diagnostics, finite true and gradient
objectives, finite nonzero gradient/update norms, and accepted guarded updates
for both fixed-sample steps. The final penalty objective was
`-11.967228889465332`, from initial `-11.952571868896484`.
The sampler-integrated FermiNet PBC driver is implemented at
`src/solidnes/excited_states/ferminet_pbc_driver.py`, with per-state Metropolis
sampling, PBC position wrapping, paper-tangent guarded updates, diagnostics,
and checkpoint save/load helpers. Run 0068 validated this path with real PBC
local energy and two walkers per state; job `128751` completed in `00:05:44`
with exit `0:0`. Run 0069 scaled the same path to four walkers per state; job
`128752` completed in `00:04:31` with exit `0:0`.
The production backend runner is implemented at
`scripts/backends/run_ferminet_pbc_excited_driver.py`. It reads YAML driver
settings, runs in checkpoint-sized segments, writes JSON/Markdown summaries,
and resumes from saved driver checkpoints. Run 0070 used this runner for a
12-iteration, four-walker real-local-energy trajectory. Job `128758` completed
on `intelgpu80g/gpu001` in `00:14:28` with exit `0:0`, wrote checkpoints at
iterations 4, 8, and 12, and accepted all guarded updates.
The post-0070 source iteration added SGD/Adam/LAMB/direct-KFAC external-state
optimizer support, EWMA overlap-gradient scaling, shared-parameter
projection, optimizer-state/running-stat checkpointing, candidate-check
cadence, update caps, and separated optimizer-update versus sharing-projection
diagnostics. CPU cheap-local-energy smokes passed for Adam, LAMB, KFAC, the
sampler-integrated KFAC driver checkpoint roundtrip, and KFAC runner resume.
The source-update record is
`records/progress/2026-05-25_ferminet_pbc_optimizer_sharing_numeric_driver_update.md`.
Run 0071 validated the direct-KFAC path with the real PBC local-energy driver.
Job `129088` completed on `amdgpu40g/gpu006` in `00:07:51` with exit `0:0`,
using one A100 40GB GPU and 8 CPU cores. The runner completed 2/2 iterations,
wrote checkpoints at iterations 1 and 2, accepted both KFAC updates, kept
optimizer update norms near `1e-3`, and ended with objective `-7.182288`,
final state energies `[-7.621417, -8.340336]`, and final off-diagonal overlap
`-0.399647`.
Run 0073 proved that the external wrapper can allocate and checkpoint a
multi-device KFAC state, but sampled GPU utilization stayed near zero while
the Python process was CPU-bound; it was cancelled after checkpoint 30.
Run 0074 moved the method onto FermiNet's native excited-state `vmc_overlap`
architecture. The first attempt exposed FermiNet's missing PBC excited-state
Hamiltonian branch; SolidNES now supplies that branch in
`src/solidnes/backends/ferminet_pbc_hamiltonian.py`. Retry job `129219`
completed 20 native KFAC steps on four A100 40GB GPUs in `00:01:55`.
Run 0075 repeated the native path with batch1024 to avoid drawing conclusions
from too-small sample counts. Job `129240` completed on `amdgpu40g/gpu005` in
`00:01:59`; native KFAC registered per-device loss shape `float32[256,2]`,
and the final overlap matrix was recorded.
Run 0076 completed the next step: native FermiNet excited-state
`vmc_overlap` KFAC with batch4096 and 50 iterations. The build-only adapter
check passed, and the Slurm dry-run selected `amdgpu40g/gpu005`, `gpu:4`,
64 CPU cores, exclusive allocation, and a 45-minute walltime.
Run 0076 was submitted as Slurm job `129249`.
Job `129249` completed in `00:03:14` with exit `0:0`. It wrote 50 training
rows, `energy_matrix.npy`, `overlap_matrix.npy`, native excited-state summary,
and benchmark summary files. The backend window was 149 seconds, or 2.98
seconds per short-run step including startup, burn-in, and compilation.
Run 0077 completed the paper-aligned native FermiNet overlap-loss smoke:
`alpha=4.0`, `scale_by=max_gap_std`, psi-ratio clipping, median local-energy
clipping, energy-based lower-state ordering, native KFAC, batch4096, and 5
iterations. Job `129257` completed on `amdgpu40g/gpu005` with four A100 40GB
GPUs in `00:02:04` and exit `0:0`. It wrote 5 training rows plus
`energy_matrix.npy`, `overlap_matrix.npy`, `overlap_gradient_scale.npy`, and
`overlap_state_ordering.npy`.
Run 0078 completed the longer 1000-step paper-aligned native FermiNet
trajectory as job `129262` on `amdgpu40g/gpu005`.
Run 0079 completed the 10000-step cleaned native method trajectory as job
`129272` on `amdgpu40g/gpu005` after fixed-shape EWM overlap-scale arrays were
added to keep KFAC traces equivalent across steps.
The native overlap-scale EWM has since been changed from a simple recursive
mean to the DeepQMC-style finite-buffer EWM. `overlap_penalty=4.0` remains a
fixed scalar alpha; the automatic component is the overlap-gradient scale.
```

## Next Concrete Action

```text
Run `0079` completed as job `129272` on `amdgpu40g/gpu005`, with `gpu:4`, 64
CPU cores, exclusive allocation, and exit `0:0`. It wrote 10000 training rows,
all native excited-state diagnostics, and a native summary. Final energy was
`-74.583840`; final state energies were `[-74.792580, -74.169594]`; final
symmetric overlap matrix was `[[1.0, 0.0159107], [0.0159107, 1.0]]`; final
overlap penalty matrix was `[[1.0, 0.000253152], [0.000253152, 1.0]]`.

Next engineering step: do not launch a 10000-step spin run from the tested beta
values. Run a smaller stability sweep around `beta <= 0.02`, likely paired with
a reduced KFAC norm constraint or learning rate, then pick a production setting.
```

## Active Or Pending Jobs

```text
None.
```

## Evidence Already Available

```text
DeepSolid reproduction:
  Job 127816
  Tail-2000 mean: -75.4161279970 Ha
  Paper reference: -75.4009 Ha

FermiNet reproduction:
  Training job 127898
  Evaluation job 127992
  Fixed-parameter evaluation mean: -75.4125655570 Ha
  Paper reference: -75.4009 Ha

FermiNet PBC-HF pretraining:
  records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
  records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md
  GPU pretraining target/backend probes passed in runs 0047--0050.
  Training integration and matched controls were recorded in runs 0053--0062.
```

## Completion Criteria For This Small Step

This small step is complete when all of the following are true:

```text
1. The DeepQMC/Szabo-Noe reference implementation has been inspected or
   intentionally deferred with a written reason. Done:
   `docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
2. A project-owned source-audit/design note records which parts of the method
   will be ported to SolidNES/FermiNet PBC and which parts are deferred. Done:
   `docs/05_reference_projects/deepqmc_penalty_excited_states.md`.
3. The penalty-based excited-state VMC objective is implemented or scaffolded
   in reusable SolidNES/FermiNet code. Partial: backend-independent utilities
   and a minimal FermiNet PBC scaffold exist under
   `src/solidnes/excited_states/`; reusable FermiNet/JAX PBC adapter wrappers,
   a fixed-sample helper, a sampler-integrated driver, and a production runner
   with checkpoint/resume support now exist.
4. The code exposes state energies plus overlap/orthogonality diagnostics.
   Partial: scaffold-level and adapter-level state-energy/overlap/penalty
   diagnostics exist; real local-energy evaluation passed in run 0063, the
   fixed-sample training helper passed in run 0067, the sampler-integrated
   driver passed in runs 0068 and 0069, and the production runner completed a
   12-iteration trajectory in run 0070.
5. A build-only or smoke-level check proves the new code path can be imported
   and configured. Done for build-only: synthetic utility/scaffold checks,
   the FermiNet/JAX adapter build check, and the cheap-local-energy
   FermiNet/JAX penalty-term, gradient-step, and multi-step optimization
   checks passed. Done for scheduled GPU smoke: run 0063 passed the real PBC
   local-energy/Laplacian check with two external state parameter trees.
   Done for source-level training integration: the cheap-local-energy
   multi-step smoke exercises the reusable training-loop helper. Done for
   scheduled real-local-energy driver integration: runs 0068 and 0069 passed.
   Done for production-runner integration: run 0070 passed. Done for native
   FermiNet spin-penalty plumbing: run 0080 passed. Done for paper-style
   spin bare-energy reporting: run 0081 passed. Done for first grouped spin
   beta sweep: task 0082 completed, but none of the variants is
   production-ready.
6. A numbered task bundle is created only for the first build/smoke/run/analysis
   step that produces project artifacts.
7. The next concrete material/probe run is defined with explicit completion
   criteria.
8. The run outcome is recorded in tasks/TASK_LEDGER.md when a numbered task
   completes or materially changes.
```

## What To Record On Every Update

For each change of state, update only this file and then write permanent details
to a dated record when the step completes.

Use these fields:

```text
State:
  ready_to_start | configuring | dry_run_done | submitted | running |
  completed | failed | blocked

Current action:
  One sentence describing what is happening now.

Next action:
  The next concrete command or decision.

Job ID:
  Slurm job id, or None.

Run ID:
  Number from records/run_index.md, or None.

Task ledger:
  Whether tasks/TASK_LEDGER.md needs to be updated.

Expected output:
  The file or metric we are waiting for.

Completion condition:
  The exact condition that lets us move to the next step.
```

## State Transition Rule

```text
Before submitting a job:
  State -> configuring or dry_run_done

After sbatch returns a job id:
  State -> submitted

After the job starts:
  State -> running

After analysis and plots are written:
  State -> completed

If a traceback, bad scheduler request, or missing result appears:
  State -> failed or blocked, with the reason written here.
```
