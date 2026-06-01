# SolidNES Current Status

Last updated: 2026-06-01, Asia/Shanghai

This file is a project-level summary. It is not the default Codex startup
context; read `CURRENT_CONTEXT.md` first.

## Current Conclusion

Three Phase 1 enabling milestones are complete:

1. Carbon-diamond paper benchmark reproduction through both DeepSolid and
   FermiNet.
2. FermiNet PBC-HF pretraining implementation and diamond-Gamma validation for
   the current cc-pVDZ workflow.
3. DeepQMC-aligned two-state excited-state method promoted into source-code
   defaults.

The current excited-state mainline is:

```text
PsiFormer/FermiNet native vmc_overlap
fused_qkv attention
independent per-state parameter trees
merge_keys: []
diagonal MCMC trace, local energy, and overlap JVP enabled
equal overlap energy weighting
fixed state index ordering
KFAC norm state scaling disabled
```

Non-empty `merge_keys` remain available as explicit comparison branches and
should resolve to `merge_key_variant`.

## Source Anchors

```text
CURRENT_CONTEXT.md
src/solidnes/excited_state_mainline.py
src/solidnes/backends/ferminet_adapter.py
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_none_batch4096_iter10000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_merge_none_batch4096_iter10000.yaml
scripts/validation/check_excited_state_mainline_defaults.py
```

## Key Evidence

Ground-state benchmark:

```text
DeepSolid job 127816: tail-2000 mean -75.4161279970 Ha
FermiNet evaluation job 127992: evaluation mean -75.4125655570 Ha
Reference: DeepSolid supplementary diamond VMC -75.4009 Ha
```

Pretraining milestone:

```text
Runs 0047--0050: JAX PBC GTO cc-pVDZ validation
AO max abs error: 1.12e-9
Occupied-MO max abs error: 8.51e-10
Runs 0053--0062: FermiNet PBC-HF training integration and controls
```

Excited-state mainline milestone:

```text
34d6574 Set no-merge excited-state mainline
916bcc4 Record no-merge excited-state milestone
records/progress/2026-06-01_excited_state_no_merge_mainline.md
```

Validation:

```text
python scripts/validation/check_excited_state_mainline_defaults.py
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py --mainline-excited-state
SOLIDNES_BUILD_ONLY=1 python scripts/backends/run_ferminet_train.py configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_merge_layers_batch4096_iter10000.yaml
python scripts/validation/check_ferminet_native_overlap_loss_alignment.py
```

## Current Direction

Use the source-code no-merge excited-state mainline for controlled periodic
material tests. Keep `merge_keys` sweeps as labeled comparison work, not as the
default production route.

## History

Detailed history is intentionally not kept here. Use:

```text
records/progress/
records/archive/2026-06-01_context_split/
tasks/TASK_LEDGER.md
tasks/ledger/
tasks/**/README.md
```
