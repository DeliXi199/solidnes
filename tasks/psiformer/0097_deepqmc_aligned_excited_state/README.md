# Task 0097: DeepQMC-Aligned Excited-State PsiFormer

## Purpose

Rerun the two 0096 excited-state PsiFormer attention variants after moving the
native FermiNet excited-state path to the DeepQMC-style route: one complete
parameter tree per state, coupled through the overlap-penalty objective.

## Submitted Configs

- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_deepqmc_aligned_fullnode_anygpu_fused_qkv_x64_b4096_iter10000_levmap128_jaxattn.yaml`
- `configs/experiment/diamond_c_psiformer_pbc_gamma_attention_deepqmc_aligned_fullnode_anygpu_ferminet_x64_b4096_iter10000_levmap128_jaxattn.yaml`

## Alignment Checked Before Submit

- One complete network parameter tree per state: `independent_state_params: true`.
- DeepQMC-style mean-log wavefunction centering before overlap ratios.
- DeepQMC-style overlap tangent prefactor from the clipped overlap mean.
- Equal per-state overlap weights for two states: `(0.5, 0.5)`.
- Fixed state-index ordering by default: `overlap_sort_states_by: null`.
- KFAC norm constraint remains `0.001` without extra state-count scaling.

## Expected Outputs

- `train_stats.csv`
- `energy_matrix.npy`
- `overlap_matrix.npy`
- `overlap_symmetric_matrix.npy`
- `overlap_penalty_matrix.npy`
- `overlap_gradient_scale.npy`
- `overlap_state_ordering.npy`
- `overlap_scale_energy_ewm.npy`
- `overlap_scale_std_ewm.npy`

## Submitted Jobs

Submitted on 2026-05-29 CST with the same full-node policy used for the 0096
x64 excited-state comparison: `amdgpu40g,amdgpu80g`, 4 GPUs, 64 CPU cores,
exclusive node, and a 3 day time limit.

| Variant | Job ID | Job name | Status |
| --- | --- | --- | --- |
| fused-QKV | `133753` | `0097-psi-fused-deepqmc-10k` | canceled while diagnosing the control failure |
| FermiNet Q/K/V control | `133752` | `0097-psi-fermi-deepqmc-10k` | failed at first KFAC update |

Earlier submissions `133743` and `133744` were canceled before running because
they used the DeepQMC overlap-loss alignment but still shared network parameters
between states.

Failure diagnosis:

- `133752` built the independent-state network and completed MCMC burn-in.
- The first KFAC curvature update failed with
  `ValueError: (2, 256) is incompatible with (1024, 2, 2, 12, 256)`.
- Root cause: the first independent-state wrapper stacked every parameter leaf
  along state axis 0, which made KFAC see layer-norm parameters such as
  `(2, 256)`. KFAC's scale/shift curvature block does not know how to reduce
  the extra state/output axes in the corresponding activations.
- Fix: keep one full parameter pytree per state instead of stacked parameter
  leaves. This preserves DeepQMC-style independent state parameters while
  exposing ordinary single-state parameter leaves to KFAC.

Retry 1 submitted after the KFAC-shape fix on 2026-05-29 CST:

| Variant | Job ID | Job name | Status after submit |
| --- | --- | --- | --- |
| fused-QKV | `133788` | `0097-psi-fused-deepqmc-r1` | completed, analyzed |
| FermiNet Q/K/V control | `133789` | `0097-psi-fermi-deepqmc-r1` | completed, analyzed |

Plans:

- `outputs/slurm_plans/plan_deepqmc_aligned_fused_qkv_x64_b4096_i10000_submit.json`
- `outputs/slurm_plans/plan_deepqmc_aligned_ferminet_x64_b4096_i10000_submit.json`
- `outputs/slurm_plans/plan_deepqmc_aligned_fused_qkv_x64_b4096_i10000_retry1_submit.json`
- `outputs/slurm_plans/plan_deepqmc_aligned_ferminet_x64_b4096_i10000_retry1_submit.json`

## 133788 Result Summary

The fused-QKV retry completed 10000 iterations on 2026-05-30 CST and wrote the
required final checkpoint `qmcjax_ckpt_009999.npz`.

Run metadata:

- Runtime: `22454.4784` seconds.
- Seconds per iteration: `2.24544784`.
- Final scalar energy: `-75.3255139 Ha`.
- Tail1000 scalar energy mean: `-75.3098049 Ha`.
- Final state energies: `[-75.4400580, -75.2109751] Ha`.
- Final fixed-order gap: `0.22908296 Ha` (`6.23366492 eV`).
- Tail1000 fixed-order gap mean: `0.17746529 Ha` (`4.82907668 eV`).
- Final symmetric overlap off-diagonal: `-8.13371004e-4`.
- Tail1000 absolute symmetric overlap off-diagonal mean: `0.00529441`.
- Final overlap-penalty off-diagonal: `6.61572390e-7`.

Artifacts:

- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_analysis.md`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_summary.json`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_timeseries.csv`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_iteration_evolution_{full,after1000,last1000}.png`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_state_energy_gap_{full,after1000,last1000}.png`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_overlap_evolution_{full,after1000,last1000}.png`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_ground_excited_after2000_roll1000.png`
- `results/validation/133788_fused_qkv/psiformer_0097_deepqmc_133788_gap_after2000_roll1000.png`

## 133789 Result Summary

The FermiNet Q/K/V control retry completed 10000 iterations on 2026-05-30 CST
and wrote the required final checkpoint `qmcjax_ckpt_009999.npz`.

Run metadata:

- Runtime: `22016.8048` seconds.
- Seconds per iteration: `2.20168048`.
- Final scalar energy: `-75.3391858 Ha`.
- Tail1000 scalar energy mean: `-75.3049075 Ha`.
- Final state energies: `[-75.4491703, -75.2292012] Ha`.
- Final fixed-order gap: `0.21996915 Ha` (`5.98566562 eV`).
- Tail1000 fixed-order gap mean: `0.19280693 Ha` (`5.24654385 eV`).
- Final symmetric overlap off-diagonal: `0.0`.
- Tail1000 absolute symmetric overlap off-diagonal mean: `0.00606315`.
- Final overlap-penalty off-diagonal: `0.0`.

Artifacts:

- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_analysis.md`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_summary.json`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_timeseries.csv`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_iteration_evolution_{full,after1000,last1000}.png`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_state_energy_gap_{full,after1000,last1000}.png`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_overlap_evolution_{full,after1000,last1000}.png`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_ground_excited_after2000_roll1000.png`
- `results/validation/133789_ferminet_qkv/psiformer_0097_deepqmc_133789_gap_after2000_roll1000.png`

## Retry-1 Variant Comparison

Both retry-1 jobs completed and are finite. The FermiNet Q/K/V control was
slightly faster in this DeepQMC-aligned independent-state route:

- `133788` fused-QKV: `2.24544784 s/iter`.
- `133789` FermiNet Q/K/V: `2.20168048 s/iter`.
- FermiNet Q/K/V was about `1.99%` faster for the full training run.

Late-window energetics differ modestly:

- Tail1000 scalar energy mean: fused-QKV `-75.3098049 Ha`; FermiNet Q/K/V
  `-75.3049075 Ha`.
- Tail1000 fixed-order gap mean: fused-QKV `4.82907668 eV`; FermiNet Q/K/V
  `5.24654385 eV`.
- Tail1000 absolute symmetric-overlap mean: fused-QKV `0.00529441`; FermiNet
  Q/K/V `0.00606315`.

Comparison plots:

- `results/validation/comparison/psiformer_0097_deepqmc_retry1_ground_excited_after2000_roll1000_comparison.png`
- `results/validation/comparison/psiformer_0097_deepqmc_retry1_gap_after2000_roll1000_comparison.png`

The plotting scripts also emit SVG copies locally; those are intentionally left
untracked.
