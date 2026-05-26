# FermiNet PBC Direct-KFAC Real-Local-Energy Driver Smoke

Date: 2026-05-25

## Summary

Implemented and validated the next direct-KFAC iteration for the sampler-integrated
FermiNet PBC excited-state driver.

Source-side update:

- KFAC now receives step-specific precomputed penalty terms through the KFAC
  batch instead of capturing them in the optimizer objective closure.
- `kfac_jax.Optimizer` objects are cached per adapter/config/shape signature, so
  repeated driver steps can reuse the same optimizer object while checkpoints
  continue to store only serializable native KFAC state.
- The batch-size extractor and KFAC sample packing support the new nested batch
  layout.

Scheduled validation:

- Task: `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke`
- Config:
  `configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_smoke.yaml`
- Job: `129088`
- Partition/node: `amdgpu40g/gpu006`
- Resources: one A100 40GB GPU, 8 CPU cores
- Slurm status: `COMPLETED`, exit `0:0`, elapsed `00:07:51`

Run result:

- External states: 2
- Walkers per state: 2
- Completed iterations: 2/2
- Optimizer: direct KFAC
- Local energy source: real PBC local energy/Laplacian
- Initial penalty objective: `-6.8448691368`
- Final penalty objective: `-7.1822881699`
- Final state energies: `[-7.6214170456, -8.3403358459]`
- Final off-diagonal overlap: `-0.3996469676`
- Accepted KFAC updates: 2/2
- Optimizer update norms: `0.0010000002`, `0.0010000000`
- Checkpoints: `driver_iter_000001.pkl`, `driver_iter_000002.pkl`

Validation commands run locally before submission:

```bash
python -m py_compile src/solidnes/excited_states/ferminet_pbc_training.py \
  src/solidnes/excited_states/ferminet_pbc_driver.py \
  scripts/backends/run_ferminet_pbc_excited_driver.py \
  scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py \
  scripts/validation/check_ferminet_pbc_driver_smoke.py

conda run -n solidnes-ferminet-jax0101-cuda12 env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_penalty_opt_smoke.py \
  --platform cpu --walkers 2 --steps 2 --optimizer kfac \
  --overlap-ewma-decay 0.5 --param-share-keys layers/streams \
  --candidate-check-period 2 --learning-rate 1e-4 \
  --max-grad-l2-norm 1000 --max-update-l2-norm 1e-3 \
  --kfac-damping 0.001 --kfac-norm-constraint 0.001 --kfac-invert-every 1

conda run -n solidnes-ferminet-jax0101-cuda12 env PYTHONPATH=external/ferminet:src \
  python scripts/validation/check_ferminet_pbc_driver_smoke.py \
  configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_smoke.yaml \
  --platform cpu --local-energy-source cheap --iterations 1 --walkers 2
```

Primary artifacts:

- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/results/validation/ferminet_pbc_driver_run_summary.md`
- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/results/validation/ferminet_pbc_driver_run_summary.json`
- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/outputs/slurm_plans/plan.json`
- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/logs/slurm/solidnes-0071-nes-kfac_129088.log`

## Next Step

Use the checkpointed direct-KFAC path for a longer controlled real-local-energy
trajectory. Increase walkers per state only after confirming the single-GPU KFAC
memory/runtime envelope, or add multi-device KFAC support if the larger run needs
more than one GPU.
