# SolidNES Current Status

Last updated: 2026-06-01, Asia/Shanghai

## Current Conclusion

Three Phase 1 enabling milestones are complete at the project-validation level:

1. The carbon-diamond paper benchmark reproduction is complete through both
   DeepSolid and FermiNet.
2. The FermiNet PBC-HF pretraining implementation and diamond-Gamma validation
   milestone is complete for the current cc-pVDZ workflow.
3. The DeepQMC-aligned two-state excited-state method is now the SolidNES
   source-code mainline: PsiFormer/FermiNet native `vmc_overlap`, independent
   per-state parameter trees, `merge_keys: []` by default, diagonal
   MCMC/local-energy/overlap-JVP paths, equal overlap energy weighting, fixed
   state ordering, and KFAC norm state scaling disabled.

The project can now move from ground-state and pretraining route hardening to
the next phase: use the source-code mainline excited-state route for controlled
periodic material tests and keep non-empty `merge_keys` only as explicit
comparison branches.

The source milestone was committed and pushed as:

```text
34d6574 Set no-merge excited-state mainline
```

Validation evidence:

```text
python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_layers_batch4096_iter10000.yaml
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
```

The no-merge config resolves to `excited_state_route_role: mainline` with
`independent_state_merge_keys: ()`. A non-empty merge config resolves to
`excited_state_route_role: merge_key_variant`, confirming that merge support is
still present but not the default.

Current excited-state implementation status: the native FermiNet PBC
`vmc_overlap` path now includes the paper-aligned overlap loss cleanup,
DeepQMC-style overlap-scale EWM, KFAC state-count norm scaling, symmetric
overlap diagnostics, and optional spin penalty plumbing through FermiNet
`spin_energy` with `s2_matrix.npy` diagnostics. Spin-penalized runs now write
`bare_energy_matrix.npy` so training energies and physical Hamiltonian energies
are separated as in the paper workflow. The spin path passed GPU smoke runs
`0080` and `0081`. The first grouped 1000-step beta sweep is task `0082`, with
variants under `runs/beta002` through `runs/beta020`; it found no
production-ready setting at the current optimizer parameters: `beta=0.02` was
finite but noisy, `beta=0.05` went NaN, and `beta=0.10/0.20` produced finite
but overly large bare gaps. Task `0086`, the grouped 12-point 2000-step
follow-up sweep over `0.000--0.030`, completed on `amdgpu80g/gpu002` with all
final jobs exiting `0:0`. It also found no production-ready beta at the current
KFAC settings. `beta=0.008` is the least bad continuation candidate, while
`beta=0.002` and `beta=0.018` are useful controls; `beta=0.012` and
`beta=0.030` had transient non-finite `S^2`/bare-energy diagnostic frames.

## Carbon-Diamond Benchmark

Both routes reproduce the DeepSolid supplementary diamond VMC reference energy
and reach equal or lower energies:

```text
DeepSolid supplementary diamond VMC reference: -75.4009 Ha
```

### DeepSolid Direct Route

The direct DeepSolid calculation reached the paper result.

Evidence:

```text
Job ID: 127816
Backend: DeepSolid
Setup: C diamond, cc-pVDZ, KFAC, batch4096, MCMC20
Resources: 4 GPU + 64 CPU
Elapsed: 03:30:43
Rows: 10000, steps 1000--10999
Last energy: -75.4010653122 Ha
Last-100 mean: -75.4192298683 Ha
Tail-2000 mean: -75.4161279970 Ha
Tail block stderr: 0.0006453174 Ha
Mean pmove: 0.5397935999
Final variance: 0.5978799387
```

Interpretation:

```text
The last energy already matches/slightly beats -75.4009 Ha, and the robust
tail-2000 mean is lower by about 15.23 mHa. This is enough to count the
DeepSolid direct route as a successful benchmark reproduction.
```

Record:

```text
records/progress/2026-05-22_carbon_diamond_ccpvdz_paper_kfac_probe.md
tasks/phase1_diamond_c/ccpvdz/continuation/0029_deepsolid_ccpvdz_paper_kfac_continue_ckpt999_batch4096_mcmc20_iter11000/results/validation/training_summary.md
```

### FermiNet Route

The FermiNet x64 route reached and exceeded the same paper result, with an
independent fixed-parameter evaluation.

Evidence:

```text
Training job: 127898
Evaluation job: 127992
Backend: FermiNet
Setup: C diamond paper geometry, x64, KFAC, FOLX Forward Laplacian,
       batch4096, MCMC20
Evaluation checkpoint: qmcjax_ckpt_018349.npz
Evaluation resources: 2 GPU + 96 CPU
Evaluation elapsed: 00:22:29
Evaluation rows: 2000
Evaluation all-row mean: -75.4125655570 Ha
Evaluation 5-block stderr: 0.0004411545 Ha
Evaluation last-1000 mean: -75.4118625314 Ha
Evaluation last energy: -75.4374770841 Ha
Mean pmove: 0.5362353088
Last ewvar: 0.0001866115
FOLX tile warnings: 0
Tracebacks: 0
```

Interpretation:

```text
The fixed-parameter evaluation mean is lower than -75.4009 Ha by about
11.67 mHa. It also agrees with the x64 training tail-10000 mean
(-75.4114059555 Ha) within about 1.16 mHa, so the FermiNet result is not just
a transient training-chain value.
```

Records:

```text
records/progress/2026-05-23_ferminet_x64_paper_pilot20k_submit.md
records/progress/2026-05-23_ferminet_x64_eval_ckpt18349_iter2000_submit.md
records/progress/2026-05-23_task_bundle_migration.md
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/results/validation/benchmark_summary.md
tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/
tasks/phase1_diamond_c/pbc_gamma/evaluation/0046_ferminet_x64_eval_ckpt18349_batch4096_mcmc20_iter2000/
```

## FermiNet PBC-HF Pretraining

The FermiNet PBC-HF pretraining route is implemented, GPU-tested, and usable
for the current carbon-diamond Gamma, cc-pVDZ workflow.

Evidence:

```text
Implementation record: records/progress/2026-05-23_ferminet_pbc_hf_pretraining.md
Milestone record: records/progress/2026-05-24_ferminet_pbc_hf_pretraining_milestone.md

PySCF PBC-HF target benchmark:
  Run 0047
  Loss: 2.14205 -> 0.0179985
  Steady mean step: 0.058384 s

JAX PBC GTO cc-pVDZ validation:
  Run 0050
  Image cutoff: 3
  AO max abs error: 1.12e-9
  Occupied-MO max abs error: 8.51e-10
  Mean step: 0.023191 s
  Speedup vs PySCF cc-pVDZ target benchmark: about 2.52x total step

Training integration:
  Runs 0053, 0054: pretrain + KFAC/FOLX completed without traceback
  Runs 0055, 0056: matched no-pretrain 1000-step controls
  Runs 0057, 0058: timeboxed no-pretrain controls
  Runs 0059--0062: timeboxed pretrain-length sweep
```

Interpretation:

```text
PBC-HF pretraining is now a working, instrumented FermiNet option for diamond
Gamma cc-pVDZ. It improves fixed-iteration early training in the 1000-step
comparison, but the matched short wall-clock comparisons are mixed. It should
be kept as a validated option, not treated yet as a universal production
default.
```

## Current Task State

As of 2026-06-01, task `0103` completed the attention x merge-key comparison
bundle and the project mainline has been updated to the no-merge DeepQMC-aligned
route. The source-of-truth constants in
`src/solidnes/excited_state_mainline.py` now point at
`configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_none_batch4096_iter10000.yaml`
and
`configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_batch4096_iter10000.yaml`.
The older shared-`layers` path remains available through explicit merge-key
configs for controls.

Task `0096` completed the formal PsiFormer attention speed comparison. It kept
pretraining out of scope and ran the paper-scale native training comparison
under `tasks/psiformer/0096_psiformer_attention_full_stack/`. Full-node jobs
`131735` upstream/FermiNet attention and `131736` fused-QKV attention both ran
on `amdgpu40g/gpu006` with 4 GPUs, 64 CPU cores, batch4096, 10000 iterations,
and exit `0:0`; both disabled spin penalty and S2 observables. Runtime
metadata reports 0.514879 s/iteration for upstream attention and 0.518262
s/iteration for fused-QKV, so fused-QKV was 0.657% slower end-to-end in the
current native KFAC/FOLX training path. Earlier fixed-partition submissions
`131692`--`131695` were redundant, and the 2000-step combined submissions
`131697`--`131698` were replaced by the completed 10000-step pair.
Per-step state-energy/gap plots were reconstructed from `energy_matrix.npy`.
The final single-step gap is 8.088 eV for upstream and 4.685 eV for fused-QKV,
while the last-1000 mean gap is 5.333 eV for upstream and 7.932 eV for
fused-QKV; fused-QKV swaps state ordering for most late steps, so the report
uses tail and rolling statistics alongside the final row.

Those first 10000-step comparison jobs used the speed precision profile. On
2026-05-28, matching fp64/no-TF32 reruns were added and submitted with
`runtime.precision_profile=fp64`, `runtime.x64_enabled=true`,
`JAX_ENABLE_X64=1`, and `psiformer.tf32=false`. Job `131952` is the upstream
attention fp64 rerun and job `131953` is the fused-QKV fp64 rerun; both keep
batch4096, 10000 iterations, 4 GPUs, 64 CPU cores, no spin penalty, and no S2
observables. Upstream job `131952` reached FOLX graph construction and showed a
PsiFormer spin-feature concatenate sparse-mask warning, so the first x64 attempt
`131952`/`131953` was cancelled/replaced. SolidNES now patches the PsiFormer
input layer so the fixed spin channel is added through a zero-derivative
coordinate-dependent term before concatenation, preserving values while avoiding
that FOLX x64 fallback in the minimal repro. Clean FOLX-fix jobs are `131974`
fused-QKV attention and `131975` upstream/FermiNet attention.

Task `0095` is the latest completed PsiFormer native training smoke. It
validates the native training path after task `0094` added
`model.attention.implementation: auto | ferminet | fused_qkv`. Task `0095`
ran build/config checks, a tiny `auto` GPU smoke, a matched batch512
upstream-vs-fused-QKV training comparison, and a fused-QKV batch1024 timing
probe under `tasks/psiformer/0095_psiformer_native_training_smoke/`.
Jobs `131661`, `131664`, `131666`, and `131667` all completed on
`test/test001` with exit `0:0`. The GPU `auto` policy resolved to
`fused_qkv`, and KFAC registration confirmed the fused `qkv_w` block.

The short full-training b512 comparison was effectively tied and slightly
slower for fused-QKV: upstream FermiNet attention took 36.855 s/iteration and
fused-QKV took 37.075 s/iteration (`0.994x`). This means the five-step native
training smoke is dominated by KFAC/local-energy/MCMC/JIT overhead rather than
attention projection launch count. The task `0094` forward-only CUDA benchmark
still supports the GPU default: for 256 walkers on `test001`/RTX 4090,
upstream median forward time was 0.000454 s and fused-QKV was 0.000432 s, a
1.051x median speedup with exact output agreement. Since future PsiFormer
calculations are GPU-only, `auto` now resolves directly to `fused_qkv`;
`ferminet` remains only as an explicit ablation/control option.

That 0096 attention validation is now superseded by the 0102/0103 DeepQMC
alignment and merge-key comparison work for default method selection.

Task `0088` completed the previous excited-state step. It is the requested
long beta=0 native FermiNet PBC two-state `vmc_overlap` baseline with 100000
iterations, batch4096, KFAC, overlap alpha 4.0, `max_gap_std` scaling,
`spin_penalty=0.0`, and `S^2` diagnostics enabled. Slurm job `129450`
completed on `amdgpu80g/gpu002` with exit `0:0` in `03:46:24`. It wrote all
100000 rows. Final scalar energy is `-75.037605 Ha`; final state energies are
`[-75.096855, -74.919098] Ha`; final gap is `4.837 eV`; tail200 gap median is
`9.205 eV`. The spin diagnostic is not controlled: final `S^2` diagonal/trace
is `[1.4585, 81.4612] / 82.9197`; full-run `S^2` diagnostics contain
35 non-finite frames; the last 10000 finite frames include 139 frames with
`|S^2 trace| > 10` and 34 frames with `|S^2 trace| > 50`.

The excited-state penalty-VMC route now has backend-independent overlap and
penalty utilities, a reusable FermiNet PBC external-state adapter, and
build-only cheap-local-energy checks for penalty terms, one gradient step, and
three consecutive paper-tangent guarded updates. Run `0063` passed the first
scheduled real PBC local-energy/Laplacian smoke on `intelgpu80g/gpu001`; the
full-node retry job `128439` used both A100 80GB GPUs, 96 CPU cores, and an
exclusive allocation. The reusable fixed-sample external-state training-loop
helper now has ordered lower-state stop-gradient overlap behavior, psi-ratio
clipping, local-energy clipping, automatic overlap-gradient scaling, and
finite-gradient/update/candidate-term guards before parameter commits. The
cheap-local-energy validation stack passes after this source change.

Run `0064`, the first scheduled real-local-energy multi-step training-loop
smoke, failed method-side because the previous direct real-local-energy
`value_and_grad` plus plain SGD path produced non-finite final local energies.
Run `0065` then showed the first paper-tangent implementation still let a
non-finite surrogate contaminate the gradient-objective forward value. That is
now fixed by evaluating true penalty terms outside `value_and_grad`, reusing
precomputed local-energy values in the paper-tangent gradient objective, and
using a forward-safe zero-primal surrogate correction.

Run `0066` confirmed the NaN objective failure was fixed but exposed a
degenerate one-walker smoke condition: with seed `47`, the centered
score-function energy tangent and overlap tangent gave zero gradient. Run
`0067` reran the fixed smoke with two walkers per state and passed. Job
`128677` completed on `intelgpu80g/gpu001` in `00:05:04` with exit `0:0`,
using both A100 80GB GPUs and 96 CPU cores. The validation summary recorded
finite `[2, 2]` real local-energy matrices, finite state energies, finite
overlap diagnostics, finite true and gradient objectives, accepted guarded
updates at both steps, and nonzero gradient norms `557.018` and `555.336`.

The sampler-integrated FermiNet PBC excited-state driver is now implemented and
validated on scheduled GPUs. Run `0068` passed the first real-local-energy
driver smoke with two walkers per state; job `128751` completed in `00:05:44`
with exit `0:0`, sampler acceptance `0.75` and `0.75`, and checkpoint
roundtrip `1233826` bytes. Run `0069` scaled the same path to four walkers per
state; job `128752` completed in `00:04:31` with exit `0:0`, sampler
acceptance `0.791667` and `0.75`, and checkpoint roundtrip `1234734` bytes.

The next step is production-runner integration for the excited-state path:
configured checkpoint persistence, resume from driver checkpoints, and a first
longer controlled trajectory under the next task bundle.

That step is now complete. The production runner
`scripts/backends/run_ferminet_pbc_excited_driver.py` supports YAML-driven
driver runs, checkpoint-sized segments, checkpoint resume, and JSON/Markdown
trajectory summaries. Run `0070` completed a 12-iteration real-local-energy
controlled trajectory with four walkers per state. Job `128758` completed on
`intelgpu80g/gpu001` in `00:14:28` with exit `0:0`, wrote checkpoints at
iterations 4, 8, and 12, and accepted all guarded updates. The final penalty
objective was `-13.3468618393`, with final state energies
`[-17.5776462555, -11.4508419037]` and final overlap off diagonal
`0.4831940234`.

The post-0070 source iteration is also complete. The external-state path now
supports SGD, Adam, LAMB, and direct KFAC through `kfac_jax.Optimizer`; carries
EWMA state-energy/std statistics for overlap-gradient scaling; projects
selected shared parameter leaves; separates optimizer-update norm from
shared-parameter projection norm; checkpoints optimizer state and running
stats; and supports candidate-check cadence plus update caps in the runner and
validation scripts. CPU cheap-local-energy smokes passed for Adam, LAMB, KFAC,
the sampler-integrated KFAC driver checkpoint roundtrip, and KFAC runner
resume. Record:
`records/progress/2026-05-25_ferminet_pbc_optimizer_sharing_numeric_driver_update.md`.

Run `0071` completed the scheduled direct-KFAC real-local-energy driver smoke.
Job `129088` ran on `amdgpu40g/gpu006` with one A100 40GB GPU and 8 CPU cores,
completed in `00:07:51` with exit `0:0`, wrote checkpoints at iterations 1
and 2, accepted both KFAC updates, kept optimizer update norms at about
`1e-3`, and ended with objective `-7.1822881699`, final state energies
`[-7.6214170456, -8.3403358459]`, and final overlap off diagonal
`-0.3996469676`.

The direct-KFAC bridge has been upgraded for multi-device execution. KFAC now
uses FermiNet's `pmap` axis name, replicates parameters and native optimizer
state across local devices, shards walker batches over the walker axis, stores
native-state metadata for checkpoint/resume compatibility, and skips the
previous redundant single-device outer `value_and_grad` on KFAC steps.

Run `0072`, the first 100-iteration full-node trial, was cancelled after
`00:12:02` because it predated this multi-device KFAC upgrade. Run `0073`
superseded it with eight walkers per state and external-wrapper multi-device
KFAC. It reached checkpoints 10/20/30, but sampled GPU utilization stayed near
zero while the Python process was CPU-bound, so it was cancelled after
`00:23:45`.

The source-level direction then changed from external-wrapper optimization to
native FermiNet integration. The SolidNES FermiNet config adapter now supports
native excited-state settings (`cfg.system.states`, `cfg.optim.objective`,
overlap penalty, and overlap weights). The SolidNES PBC Hamiltonian wrapper now
supplies the missing excited-state PBC local-energy branch using the PBC Ewald
potential and FermiNet's excited-state kinetic/local-energy matrix machinery.

Run `0074` validated this native path. The first job exposed the upstream PBC
Hamiltonian `NotImplementedError`; after the SolidNES Hamiltonian branch was
added, retry job `129219` completed 20 native `vmc_overlap` KFAC steps on four
A100 40GB GPUs in `00:01:55`. It wrote `train_stats.csv` and
`energy_matrix.npy`; the final loss was `-22.488228`, mean pmove was
`0.910938`, and the final state-energy vector was
`[-22.203577, -23.186222]`.

Run `0075` addressed the small-sample concern by repeating the native path with
batch1024. Job `129240` completed on `amdgpu40g/gpu005` in `00:01:59` with
exit `0:0`; native KFAC registered loss shape `float32[256,2]` per device over
four GPUs. The run wrote `train_stats.csv`, `energy_matrix.npy`, and the new
`overlap_matrix.npy`; the final loss was `-22.453096`, final EW mean was
`-22.400412`, mean pmove was `0.911453`, the final state-energy vector was
`[-22.497761, -22.392517]`, and the final overlap matrix was
`[[1.0, 0.0773164], [0.123951, 1.0]]`.

Run `0076` then moved to batch4096 for a short speed/stability baseline.
Job `129249` completed on `amdgpu40g/gpu005` in `00:03:14` with exit `0:0`.
JAX saw all four A100 40GB GPUs; native KFAC registered per-device loss shape
`float32[1024,2]`; 50 rows were written. The final loss was `-24.691084`,
final EW mean was `-24.117025`, mean pmove was `0.910310`, the final
state-energy vector was `[-25.515295, -23.108868]`, and the final overlap
matrix was `[[1.0, 0.0672966], [0.328056, 1.0]]`. The backend window was 149s
or `2.98 s/step`, but this 50-step number still includes startup, burn-in, and
compilation overhead, so a longer native run is needed for a fair steady-state
comparison with the ground-state FermiNet baseline.

The native overlap loss has now been aligned with the key Szabo-Noe/DeepQMC
settings in the FermiNet `vmc_overlap` path: `alpha=4.0`,
`scale_by=max_gap_std`, psi-ratio clipping, median local-energy clipping, and
energy-based lower-state ordering for the detached upper-triangle overlap
tangent. Run `0077` validated this paper-aligned path as a short GPU smoke on
`amdgpu40g/gpu005` with four A100 40GB GPUs, native KFAC, batch4096, and 5
iterations. Job `129257` completed in `00:02:04` with exit `0:0`; final energy
was `-22.351885`, final state-energy vector was
`[-22.548399, -21.983301]`, final overlap matrix was
`[[1.0, 0.0314396], [0.0647455, 1.0]]`, final overlap-gradient scale was
`[[5.0, 5.0], [5.0, 5.0]]`, and final state ordering was `[0, 1]`.

Run `0078` completed the longer paper-aligned native FermiNet two-state
trajectory on `amdgpu40g/gpu005` with four A100 40GB GPUs, native KFAC,
batch4096, and 1000 iterations. Job `129262` completed in `00:03:40` with
exit `0:0`. It wrote 1000 rows and all native diagnostics. Final energy was
`-73.959910`, final EW mean was `-73.935610`, final EW variance was
`0.00348556`, tail-100 mean energy was `-73.867186`, final state-energy vector
was `[-74.176765, -73.526321]`, final overlap matrix was
`[[1.0, 0.0186032], [-0.0343843, 1.0]]`, final overlap-gradient scale was
`[[5.0, 5.0], [5.0, 5.0]]`, and final state ordering was `[0, 1]`.

Run `0079` completed the method-cleanup 10000-step native FermiNet two-state
trajectory after keeping spin penalty out of scope. The native path now has a
SolidNES `szabo_noe_2024_penalty` method profile, EWM overlap-gradient
scale/order inputs, state-count KFAC norm scaling, fixed-shape KFAC trace data,
and separate raw/symmetric/penalty overlap diagnostics. First submit job
`129267` failed after step 0 because KFAC saw non-equivalent traces when EWM
fields changed from `None` to arrays; after fixed-shape NaN EWM initialization,
retry job `129272` completed on `amdgpu40g/gpu005` in `00:20:11` with exit
`0:0`. It wrote 10000 rows. Final energy was `-74.583840`, final EW mean was
`-74.684850`, final EW variance was `0.00702351`, mean pmove was `0.549417`,
final state-energy vector was `[-74.792580, -74.169594]`, final symmetric
overlap matrix was `[[1.0, 0.0159107], [0.0159107, 1.0]]`, and final overlap
penalty matrix was `[[1.0, 0.000253152], [0.000253152, 1.0]]`.

After rechecking the DeepQMC source, the overlap-scale EWM was aligned more
closely to the reference implementation: `overlap_penalty=4.0` remains the
fixed scalar alpha, while the automatic component is overlap-gradient scaling
from `energy_ewm/std_ewm`. SolidNES now uses the DeepQMC-style finite-buffer EWM
weights instead of the previous simple recursive mean.

## Milestones

### Completed

1. Project scaffold and task records are in place.
2. DeepSolid environment and GPU execution path were made usable on modern JAX.
3. Carbon-diamond DeepSolid smoke, continuation, and validation runs completed.
4. Direct DeepSolid paper-like KFAC/cc-pVDZ run reached the paper diamond VMC
   reference.
5. FermiNet PBC diamond scaffold was built on latest JAX.
6. FOLX Forward Laplacian was enabled and benchmarked; long KFAC runs showed
   clear speed benefit after overhead is amortized.
7. FermiNet x64 paper-geometry run reached the paper diamond VMC reference.
8. FermiNet fixed-parameter evaluation confirmed the x64 training result.
9. True PBC-HF pretraining support for the FermiNet PBC route was added.
10. JAX PBC GTO pretraining target evaluation was validated for diamond Gamma
    `ccpvdz` and benchmarked faster than the host-side PySCF target path.
11. FermiNet PBC-HF pretraining completed GPU integration checks and matched
    no-pretrain comparisons for the current diamond Gamma workflow.
12. Numbered task bundles, `tasks/TASK_LEDGER.md`, and the retired top-level
    generated artifact cleanup are in place.

### Current Milestone

```text
Completed:
- Diamond ground-state benchmark reproduction for both:
  - DeepSolid direct calculation
  - FermiNet calculation
- FermiNet PBC-HF pretraining implementation and diamond-Gamma validation
- Native FermiNet PBC overlap-penalty excited-state method cleanup and a
  10000-step two-state trajectory, excluding spin penalty
```

The next milestone is to use the 0079 artifacts for physical/convergence review
and then choose concrete material tests with explicit direct-gap, indirect-gap,
twist, and finite-size caveats.

## Important Caveats

1. DeepSolid job `127816` is a training-chain statistic, not yet a separate
   fixed-checkpoint evaluation. It is already strong enough for the current
   milestone, but a fixed-checkpoint DeepSolid evaluation would make the
   comparison cleaner.
2. FermiNet evaluation job `127992` restored `qmcjax_ckpt_018349.npz`, because
   the 20k training job did not write a final `qmcjax_ckpt_019999.npz`.
3. The current FermiNet result is already strong, but a longer fixed evaluation
   can reduce the statistical error bar if needed for a formal report.
4. PBC-HF pretraining is validated for the current diamond Gamma `ccpvdz`
   workflow only. Non-Gamma twists, other cells, pseudopotentials, or larger
   basis sets still require AO/MO validation.
5. PBC-HF pretraining improves fixed-iteration early training in the tested
   1000-step comparison, but short wall-clock timeboxed comparisons are mixed.
   It is not yet a universal production default.

## Next Phase

The next main task is excited-state method reproduction and material testing.

Recommended next work items:

1. Continue the Szabo and Noe JCTC 2024 penalty-based excited-state VMC work
   from the native FermiNet PBC overlap-penalty path now implemented; spin
   penalty remains deferred.
2. Keep the first target narrow: carbon diamond primitive cell, Gamma point,
   same ground-state geometry and basis, and clear orthogonality/overlap
   diagnostics.
3. Use `tasks/excited_state_nesvmc/` for future numbered implementation,
   smoke, training, evaluation, and material-test task bundles only when those
   steps produce project artifacts. Pure source audits and design notes should
   stay in `docs/` or `records/progress/`.
4. After the controlled diamond probe works, choose concrete material tests and
   record direct-gap, indirect-gap, twist, and finite-size caveats explicitly.
5. Use PBC-HF pretraining as an optional initialization path, not a mandatory
   default, until longer wall-clock evidence justifies it.
6. Preserve DeepSolid as the successful ground-state reproduction reference and
   FermiNet as the primary development framework.

## Working Decision

The working decision is:

```text
Use DeepSolid as the successful direct-paper-reproduction reference.
Use FermiNet as the primary efficiency and future-development framework.
Treat FermiNet PBC-HF pretraining as a completed enabling milestone for diamond
Gamma cc-pVDZ, with mixed wall-clock conclusions.
Focus next on reproducing the penalty-based excited-state VMC method in code
and then testing it on concrete periodic materials.
```
