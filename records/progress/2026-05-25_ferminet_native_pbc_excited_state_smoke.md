# FermiNet Native PBC Excited-State Smoke

Date: 2026-05-25

## Summary

The external excited-state wrapper was not efficient enough for the user's goal
of matching FermiNet ground-state speed. Run 0073 proved that KFAC could be
placed in a multi-device state, but sampled GPU utilization stayed near zero
while the process was CPU-bound. The 100-step external-wrapper run was stopped
after checkpoint 30.

The next implementation step moved the SolidNES path onto FermiNet's native
excited-state architecture:

- `cfg.system.states`
- `cfg.optim.objective = vmc_overlap`
- native FermiNet overlap-penalty loss
- native KFAC `multi_device=True`
- native FermiNet training loop, sampler, checkpointing, and KFAC registration
- batch1024 smoke coverage to avoid over-interpreting tiny-sample GPU
  utilization

## Source Changes

- `src/solidnes/backends/ferminet_adapter.py`
  - Added SolidNES YAML support for native FermiNet excited-state settings:
    `states`, `objective`, `overlap_penalty`, and `overlap_weights`.
  - Included these fields in adapter summaries.
- `src/solidnes/backends/ferminet_pbc_hamiltonian.py`
  - Added the missing PBC excited-state local-energy branch.
  - Uses the PBC Ewald potential plus FermiNet's excited-state kinetic/local
    energy matrix machinery.
  - Supports the FOLX state-specific diagonal local-energy path required by
    native `vmc_overlap`.
- `external/ferminet/ferminet/train.py`
  - Added `overlap_matrix.npy` checkpoint logging when native excited-state
    auxiliary data contains `mean_s_ij`.
- `scripts/validation/summarize_ferminet_native_excited_run.py`
  - Added a native-run summarizer that reads `train_stats.csv`,
    `energy_matrix.npy`, and `overlap_matrix.npy`, including appended `.npy`
    frames.

## Run 0074

Task:
`tasks/excited_state_nesvmc/0074_ferminet_native_vmc_overlap_kfac_smoke/`

Config:
`configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_smoke.yaml`

First job:

- Job `129218`
- Failed in `00:00:22`
- Reason: upstream FermiNet PBC Hamiltonian raised
  `NotImplementedError: Excited states not implemented with PBC.`

Retry:

- Job `129219`
- Completed in `00:01:55`
- Node: `amdgpu40g/gpu005`
- Resources: 4 A100 40GB GPUs, 64 CPU cores, exclusive allocation
- Rows: 20
- Final loss/energy: `-22.488228`
- Final EW mean: `-22.557020`
- Final EW variance: `0.24239741`
- Mean pmove: `0.910938`
- Final state-energy vector: `[-22.203577, -23.186222]`

Summary:
`tasks/excited_state_nesvmc/0074_ferminet_native_vmc_overlap_kfac_smoke/results/validation/native_vmc_overlap_kfac_smoke_summary.md`

## Run 0075

Task:
`tasks/excited_state_nesvmc/0075_ferminet_native_vmc_overlap_kfac_batch1024_smoke/`

Config:
`configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_batch1024_smoke.yaml`

Result:

- Job `129240`
- Completed in `00:01:59`
- Node: `amdgpu40g/gpu005`
- Resources: 4 A100 40GB GPUs, 64 CPU cores, exclusive allocation
- Batch size: `1024`
- Native KFAC registered per-device loss shape: `float32[256,2]`
- Rows: 20
- Final loss/energy: `-22.453096`
- Final EW mean: `-22.400412`
- Final EW variance: `0.037927367`
- Mean pmove: `0.911453`
- Final state-energy vector: `[-22.497761, -22.392517]`
- Final overlap matrix:
  `[[1.0, 0.0773164], [0.123951, 1.0]]`

Summary:
`tasks/excited_state_nesvmc/0075_ferminet_native_vmc_overlap_kfac_batch1024_smoke/results/validation/native_ferminet_excited_summary.md`

## Next Step

Use the native path as the production basis. The remaining work is to run a
longer native trajectory at production-like batch size, align penalty scaling
with the Szabo-Noe settings, and compare per-step wall time against the
previously reproduced ground-state FermiNet baseline.
