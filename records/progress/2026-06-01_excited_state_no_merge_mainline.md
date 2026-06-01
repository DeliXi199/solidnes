# 2026-06-01 Excited-State No-Merge Mainline

The DeepQMC-aligned excited-state route has been promoted into the SolidNES
source-code mainline.

## Decision

Default excited-state calculations now use:

```text
network: PsiFormer
attention: fused_qkv
objective: vmc_overlap
states: 2
independent_state_params: true
merge_keys: []
diagonal_mcmc_trace: true
diagonal_local_energy: true
diagonal_overlap_jvp: true
overlap weights: equal by state
KFAC norm_constraint_scale_by_states: false
```

Non-empty `merge_keys` remain implemented and configurable, but they are
comparison branches rather than the production default.

## Source Anchors

```text
src/solidnes/excited_state_mainline.py
src/solidnes/backends/ferminet_adapter.py
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_none_batch4096_iter10000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_batch4096_iter10000.yaml
scripts/validation/check_excited_state_mainline_defaults.py
```

The source-of-truth mainline config now resolves through
`--mainline-excited-state`.

## Validation

Commands run before committing:

```text
python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_layers_batch4096_iter10000.yaml
python -m compileall -q src scripts/validation/check_excited_state_mainline_defaults.py
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
git diff --check
```

Observed classification:

```text
no-merge fused_qkv config: excited_state_route_role = mainline
merge-layers fused_qkv config: excited_state_route_role = merge_key_variant
```

## Git

```text
commit: 34d6574 Set no-merge excited-state mainline
remote: origin/main
```

