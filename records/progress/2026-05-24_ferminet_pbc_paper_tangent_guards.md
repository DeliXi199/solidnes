# FermiNet PBC Paper-Tangent Penalty Guards

Date: 2026-05-24, Asia/Shanghai

## Summary

After run `0064` showed that the direct real-local-energy
`value_and_grad` plus plain SGD path could produce non-finite final local
energies, the fixed-sample FermiNet PBC penalty training path was updated before
another scheduled smoke.

The source-level update adds:

- ordered lower-state stop-gradient overlap behavior;
- psi-ratio clipping and log-ratio bounding;
- local-energy clipping for the VMC score-function tangent;
- automatic overlap-gradient scaling from state energy gap/std diagnostics;
- optional global gradient clipping;
- finite-gradient, finite-update, and candidate-term finite guards before
  committing parameters.

No numbered task bundle was allocated for this source-only update. The next
numbered bundle remains `0065`, to be allocated only for the next scheduled
real-local-energy multi-step smoke or another durable validation artifact.

## Validation

Commands passed:

```bash
PYTHONPATH=src python -m py_compile \
  src/solidnes/excited_states/overlap.py \
  src/solidnes/excited_states/penalty.py \
  src/solidnes/excited_states/ferminet_pbc_scaffold.py \
  src/solidnes/excited_states/ferminet_pbc_adapter.py \
  src/solidnes/excited_states/ferminet_pbc_training.py \
  src/solidnes/excited_states/__init__.py \
  scripts/validation/check_excited_state_penalty_objective.py \
  scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py

PYTHONPATH=src python scripts/validation/check_excited_state_penalty_objective.py
PYTHONPATH=src python scripts/validation/check_ferminet_pbc_excited_scaffold.py

conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_excited_adapter_build.py

conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_terms.py

conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_grad_step.py

conda run -n solidnes-ferminet-jax0101-cuda12 \
  env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py

git diff --check
```

The multi-step cheap-local-energy smoke used `gradient_mode: paper_tangent`,
accepted all three guarded updates, and produced finite diagnostics.

## Next Step

Prepare run `0065` as a new task bundle for the next scheduled real-local-energy
multi-step smoke. Dry-run the approved FermiNet GPU submitter first, inspect the
plan, and submit only if the requested `intelgpu80g` resources and walltime are
appropriate.
