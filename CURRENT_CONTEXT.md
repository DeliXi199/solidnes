# SolidNES Current Context

Last updated: 2026-06-01, Asia/Shanghai

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
```

Non-empty `merge_keys` remain implemented and configurable, but they are
comparison branches. They should resolve as `merge_key_variant`, not
`mainline`.

## Source Of Truth

```text
src/solidnes/excited_state_mainline.py
src/solidnes/backends/ferminet_adapter.py
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_none_batch4096_iter10000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_batch4096_iter10000.yaml
scripts/validation/check_excited_state_mainline_defaults.py
```

The CLI shortcut `--mainline-excited-state` should resolve to the no-merge
fused-QKV config above.

## Current Milestone

Committed and pushed:

```text
34d6574 Set no-merge excited-state mainline
916bcc4 Record no-merge excited-state milestone
```

Project record:

```text
records/progress/2026-06-01_excited_state_no_merge_mainline.md
records/progress/2026-06-01_context_file_split.md
```

The next available task number is:

```text
0104
```

## Validation Commands

Use these for mainline/config checks:

```text
source .venv/ferminet-jax0101-cuda12/bin/activate
python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_layers_batch4096_iter10000.yaml
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
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
