# Excited-State Default Parameter Decision

Date: 2026-06-08, Asia/Shanghai

## Decision

Use the following optimizer schedule as the default for future PsiFormer
excited-state calculations unless a task explicitly declares a new sweep or
ablation:

```text
learning_rate / eta0: 0.02
learning_rate_delay / tau: 10000.0
learning_rate_decay: 1.0
```

The reference configuration for the currently validated route is:

```text
configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
configs/train/excited_state_psiformer_pbc_native_kfac_deepqmc_spin_beta10_damp1e3_default_eta2e2_tau10000_merge_none_batch4096_iter30000.yaml
```

This reference uses FermiNet-QKV PsiFormer attention, two-state `vmc_overlap`,
independent state parameters with `merge_keys: []`, DeepQMC-style spin penalty
`beta=10`, KFAC `damping=0.001`, batch size 4096, and 30000 iterations.

## Evidence

The decision uses the combined 0113/0114 fresh-start eta/tau sweep. Data source:

```text
tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/fixed_tau_eta_comparison/
```

Final-window statistics use steps 29000-29999. Rolling plots use 1000-step
windows.

Selected setting, `eta0=0.02`, `tau=10000`:

| Metric | Mean | Std | Var |
| --- | ---: | ---: | ---: |
| E0 Ha | -75.387869519 | 0.035995758 | 0.001295695 |
| E1 Ha | -75.182837834 | 0.036058101 | 0.001300187 |
| Gap eV | 5.579196363 | 1.380652527 | 1.906201399 |
| Spin `<S^2>` | 0.008607421 | 0.000666732 | 4.445320831e-7 |
| EWVar Ha^2 | 0.000654080 | 0.000342837 | n/a |

Drift from the first 1000-step window after step 20000 to the final 1000-step
window:

| Metric | Delta |
| --- | ---: |
| E0 Ha | -0.008464352 |
| E1 Ha | -0.011429573 |
| Gap eV | -0.080687754 |
| Spin `<S^2>` | -0.000830394 |

Reason for selection: this setting gives near-best final energies while also
having lower final-window E0/E1/gap/spin fluctuation than `eta0=0.01,
tau=10000`; it also avoids the larger spin and drift seen in the `tau=15000`
candidate.

## Saved Data Snapshot

Fixed snapshot archive:

```text
records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/fixed_tau_eta_comparison_analysis_20260608.tar.gz
```

SHA256:

```text
30e8d2bdb6795e67d2e156b17b2da9875f32ba7d2fee23d479cbbde5341aceca
```

The archive contains the fixed-tau analysis directory, the plotting script, and
the default train/experiment configs.
