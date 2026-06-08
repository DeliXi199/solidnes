# SolidNES Current Context

Last updated: 2026-06-08, Asia/Shanghai

This is the default short context file for Codex and other coding agents. It is
the hot handoff surface. Older history lives in `records/progress/`,
`records/archive/`, `tasks/**/README.md`, and the ledger files.

## Current Mainline

The source-code excited-state mainline is the DeepQMC-aligned no-merge
PsiFormer/FermiNet native route:

```text
objective: vmc_overlap
network: PsiFormer
attention: fused_qkv
states: 2
independent_state_params: true
merge_keys: []
diagonal_mcmc_trace: true
diagonal_local_energy: true
diagonal_overlap_jvp: true
overlap_weights: equal by state
overlap_sort_states_by: null
kfac.norm_constraint_scale_by_states: false
spin_penalty: 0.0 by default
```

Non-empty `merge_keys` remain implemented and configurable, but they are
comparison branches. They should resolve as `merge_key_variant`, not
`mainline`.

Fixed-ground is an explicit-only branch. Do not select, configure, or submit
fixed-ground jobs unless the user directly asks for `fixed-ground`. When the
user says "two excited-state methods" in the current PsiFormer context, prefer
the two attention QKV handling routes (`fused_qkv` and upstream-shaped
`ferminet`) unless they say otherwise.

Optional spin-penalty runs now use DeepQMC-style loss-level
`beta * <S^2>`/custom-JVP semantics for `vmc_overlap`; the loss operator uses a
state-specific local S² estimator matching DeepQMC `evaluate_spin`, while the
full S² matrix remains an observable diagnostic. The initial DeepQMC-reference
beta is `10.0`; loss-level smoke `0104` and final state-specific smoke `0105`
both passed through SLURM.

The default optimizer schedule for future excited-state calculations is now:

```text
learning_rate / eta0: 0.02
learning_rate_delay / tau: 10000.0
learning_rate_decay: 1.0
kfac.damping: 0.001
```

This schedule was selected from the completed 0113/0114 fresh-start
FermiNet-QKV eta/tau sweep using final-window statistics and 1000-step rolling
mean/std/variance diagnostics. Treat it as the default unless a task explicitly
declares a new sweep or ablation.

## Source Of Truth

```text
src/solidnes/excited_state_mainline.py
src/solidnes/backends/ferminet_adapter.py
src/solidnes/excited_states/penalty.py
src/solidnes/excited_states/ferminet_pbc_adapter.py
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_none_batch4096_iter10000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_batch4096_iter10000.yaml
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
scripts/validation/check_excited_state_mainline_defaults.py
```

The CLI shortcut `--mainline-excited-state` should resolve to the no-merge
fused-QKV config above.

## Current Milestone

Current milestone:

```text
spin penalty implementation complete
default excited-state optimizer schedule selected: eta0=0.02, tau=10000, decay=1.0
default damping: 0.001
analysis snapshot saved under records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/
```

Project record:

```text
records/progress/2026-06-08_spin_penalty_default_lr_milestone.md
records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/default_parameter_decision.md
records/progress/2026-06-01_excited_state_no_merge_mainline.md
records/progress/2026-06-01_context_file_split.md
records/progress/2026-06-01_fixed_ground_explicit_only_policy.md
```

The next available task number is:

```text
0115
```

## Validation Commands

Use these for mainline/config checks:

```text
source .venv/ferminet-jax0101-cuda12/bin/activate
python scripts/validation/check_excited_state_mainline_defaults.py
python scripts/validation/check_excited_state_penalty_objective.py
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
python scripts/backends/build_ferminet_config.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_layers_batch4096_iter10000.yaml
```

Expected classification:

```text
no-merge fused_qkv: route_role = mainline
merge-layers fused_qkv: route_role = merge_key_variant
```

## Active Work Rule

Before running or submitting compute:

```text
1. Read docs/00_project_guidance/slurm_task_management.md.
2. Run build-only/no-compute verification when available.
3. Use project submitters in dry-run mode first.
4. Keep plans/logs under the active task bundle.
5. Update ACTIVE_TASK.md only with the current small step.
```

## Where To Look

Default startup should read only:

```text
AGENTS.md
CURRENT_CONTEXT.md
```

Use these only when needed:

```text
CURRENT_STATUS.md      project-level summary
ACTIVE_TASK.md         current small step only
PROGRESS.md            short rolling update
records/progress/      dated completed records
records/archive/       exact pre-split context files
records/run_index.md   compact numbered-run index
tasks/TASK_LEDGER.md   compact task-ledger index
tasks/ledger/          split readable task ledgers
tasks/**/README.md     detailed task-local records
```

Context-management policy:

```text
docs/00_project_guidance/context_management.md
```
