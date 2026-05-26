# FermiNet Fixed-Ground Overlap Beta=0 20k Submit

Date: 2026-05-26, Asia/Shanghai

## Summary

Implemented and submitted task `0089`, a fixed-ground excited-state follow-up
to the long beta=0 task `0088`.

The new path trains one paper-size x64 FermiNet state and computes the overlap
penalty against a fixed preconverged ground-state checkpoint instead of
co-optimizing ground and excited states. Spin terms are disabled:
`spin_penalty=0.0` and `observables_s2=false`.

## Code And Config

- Added FermiNet loss objective `fixed_ground_overlap`.
- Added fixed-ground checkpoint loading and optional small trainable-state
  perturbation in `external/ferminet/ferminet/train.py`.
- Added SolidNES adapter plumbing for `training.fixed_ground`.
- Added config
  `configs/experiment/diamond_c_ferminet_pbc_gamma_fixed_ground_overlap_beta0_iter20000.yaml`.
- Task bundle:
  `tasks/excited_state_nesvmc/0089_ferminet_fixed_ground_overlap_beta0_iter20000/`.

Fixed ground-state source:

```text
task: 0044
checkpoint: tasks/phase1_diamond_c/pbc_gamma/training/0044_ferminet_kfac_folx_batch4096_x64_burnin1000_iter20000_paper_pilot/results/checkpoints/qmcjax_ckpt_018349.npz
energy reference: -75.4118625314 Ha
```

## Checks

Passed before submit:

- `python -m py_compile` for modified Python files.
- FermiNet adapter build-only check.
- Fixed-ground checkpoint shape compatibility check.
- Fixed-ground custom-JVP forward/backward smoke with cheap local energy.
- `git diff --check` on modified source/config files.

Check outputs are under:

```text
tasks/excited_state_nesvmc/0089_ferminet_fixed_ground_overlap_beta0_iter20000/outputs/config_checks/
```

## Slurm

Dry-run and submit used the approved FermiNet GPU submitter.

Original submitted job:

```text
job id: 129584
partition: intelgpu80g
gres: gpu:2
cpus: 96
exclusive: yes
walltime: 06:00:00
final state: cancelled while pending
```

Per user request, this `intelgpu80g`-only job was cancelled because it remained
queued. The replacement submission uses the automatic GPU policy with at least
two GPUs and queues across normal non-LLM GPU partitions.

Replacement submitted job:

```text
job id: 129670
partitions: h200,amdgpu80g,amdgpu40g,h20
gres: gpu:2
cpus: 32
exclusive: yes
walltime: 12:00:00
initial state: PD (Resources)
```

Plan:

```text
tasks/excited_state_nesvmc/0089_ferminet_fixed_ground_overlap_beta0_iter20000/outputs/slurm_plans/plan.json
tasks/excited_state_nesvmc/0089_ferminet_fixed_ground_overlap_beta0_iter20000/outputs/slurm_plans/plan_auto_gpu_requeue.json
```

## Notes

`train_stats.csv` for this objective separates the scalar optimization
objective from the physical Hamiltonian energy. The existing `energy` column is
the objective, while `physical_energy`, `fixed_ground_overlap`,
`fixed_ground_overlap_squared`, and `fixed_ground_overlap_penalty` are logged
as separate columns.
