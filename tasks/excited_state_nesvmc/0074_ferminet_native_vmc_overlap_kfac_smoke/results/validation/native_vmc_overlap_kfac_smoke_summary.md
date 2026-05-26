# Native FermiNet Excited-State KFAC Smoke

Run: `0074_ferminet_native_vmc_overlap_kfac_smoke`

Job: `129219`

Status: completed, exit `0:0`

Elapsed: `00:01:55`

Node: `amdgpu40g/gpu005`, `gpu:4`, 64 CPU cores, exclusive allocation

Config: `configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_smoke.yaml`

## Method Path

- FermiNet native `cfg.system.states = 2`
- FermiNet native `cfg.optim.objective = vmc_overlap`
- FermiNet native KFAC optimizer with `multi_device=True`
- SolidNES PBC Hamiltonian wrapper now supplies the missing excited-state PBC local-energy branch
- FOLX state-specific local-energy path for the `vmc_overlap` loss

## Result

- `train_stats.csv` rows: 20
- Final step: 19
- Final loss/energy: `-22.488228`
- Final EW mean: `-22.55702`
- Final EW variance: `0.24239741`
- Mean pmove: `0.910937523`
- pmove range: `[0.89453125, 0.9273438]`
- `energy_matrix.npy` frames: 20
- Final state-energy vector: `[-22.203577, -23.186222]`
- `overlap_matrix.npy` was added after this run, so overlap-matrix frames are not present in 0074.

## Interpretation

This proves the project can now execute the Szabo-Noe-style excited-state
penalty objective through FermiNet's native multi-state architecture for PBC
diamond, rather than through the slower external state-wrapper path.
