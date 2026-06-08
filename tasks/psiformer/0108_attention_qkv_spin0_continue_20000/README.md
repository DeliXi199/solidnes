# 0108 Attention QKV Spin-0 Continuation

This task continues the completed 0107 spin-0 PsiFormer comparison for 20000
additional KFAC steps.

Fixed-ground is intentionally not part of this task.

## Continuation Semantics

- Restore source: 0107 final checkpoints at `qmcjax_ckpt_009999.npz`
- FermiNet restore returns `t_init = 10000`
- Training config sets `iterations: 30000`
- New steps: 10000 through 29999
- New output root: `tasks/psiformer/0108_attention_qkv_spin0_continue_20000`

The completed 0107 directories are read only as restore sources and are not
used as save paths.

## Variants

| Attention QKV route | Experiment config | Slurm name |
| --- | --- | --- |
| fused_qkv | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_fused_qkv_spin_beta10_merge_none_batch4096_iter30000_continue0108.yaml` | `solidnes-0108-fused_qkv-cont20k` |
| ferminet | `configs/experiment/diamond_c_psiformer_pbc_gamma_deepqmc_attention_ferminet_spin_beta10_merge_none_batch4096_iter30000_continue0108.yaml` | `solidnes-0108-ferminet-cont20k` |

## Fixed Settings

- Batch size: 4096
- Hardware request: 4 GPU, 64 CPU, 64 GB memory
- Precision: x64 / fp64
- Objective: `vmc_overlap`
- States: 2
- Independent per-state parameter trees: enabled
- Merge keys: none
- Spin penalty: DeepQMC-style loss-level `beta * <S^2>` with `beta=10.0`
- Spin logging: `log_spin_by_state: true`, `log_every: 1`
- Full S2 matrix observable: `observables_s2: false`

## Submitted Jobs

| Job ID | Attention QKV route | Final state | Elapsed | Node |
| ---: | --- | --- | ---: | --- |
| 136170 | ferminet | COMPLETED, exit `0:0` | `09:23:02` | `gpu006` |
| 136171 | fused_qkv | COMPLETED, exit `0:0` | `09:30:38` | `gpu006` |

Submitted on 2026-06-02 through `scripts/slurm/submit_ferminet_gpu_smoke.sh`
after build-only verification passed for both continuation configs.

## Launch Verification

- `136170` loaded the 0107 FermiNet checkpoint
  `qmcjax_ckpt_009999.npz` and reached `Step 10000`.
- `136171` had not started yet at the first post-submit check because it was
  waiting for GPU resources.

## Result Summary

Analysis concatenates the original 0107 steps `0..9999` with the 0108
continuation steps `10000..29999`.

| Variant | Last1000 ground (Ha) | Last1000 excited (Ha) | Last1000 gap (eV) | Last5000 gap (eV) | Last1000 spin state0 | Last1000 spin state1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FermiNet QKV | -75.38956332 | -75.18281067 | 5.626026 +/- 1.579027 | 5.569321 +/- 1.574981 | 0.007474 | 0.009054 |
| Fused QKV | -75.38409287 | -75.18260977 | 5.482634 +/- 1.492863 | 5.558044 +/- 1.559539 | 0.006864 | 0.008084 |

Artifacts:

- `analysis/0108_combined_result_summary.md`
- `analysis/0108_combined_energy_gap_spin_rolling_after5000_window1000.png`
- `analysis/0108_combined_energy_gap_spin_rolling_after10000_window1000.png`
- `analysis/0108_combined_energy_gap_spin_rolling_after20000_window1000.png`
- `analysis/0108_combined_gap_rolling_after20000_window1000.png`
