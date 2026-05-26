# FermiNet PBC Excited-State Driver Run

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_smoke.yaml
jax_platform: gpu
external_state_params: 2
walkers_per_state: 2
completed_iterations: 2
checkpoint_interval: 1
optimizer: kfac
learning_rate: 0.0001
kfac_damping: 0.001
kfac_momentum: 0.0
kfac_norm_constraint: 0.001
kfac_invert_every: 1
overlap_ewma_decay: 0.5
param_share_keys: ['layers/streams']
candidate_check_period: 1
local_energy_source: real_pbc
elapsed_seconds: 349.078114
initial_penalty_objective: -6.844869136810303
final_penalty_objective: -7.18228816986084
```

Checkpoints:

- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/results/checkpoints/driver_iter_000001.pkl`
- `tasks/excited_state_nesvmc/0071_ferminet_pbc_driver_kfac_real_local_energy_smoke/results/checkpoints/driver_iter_000002.pkl`

Per-iteration diagnostics:

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 1.0
penalty_objective: -7.972306251525879
state_energy: [-8.208288192749023, -9.563041687011719]
state_energy_std: [8.220809936523438, 1.7650394439697266]
overlap_gradient_scale: [[5.0, 5.0], [2.402440071105957, 2.124481201171875]]
offdiag_squared_overlap: 0.18267174065113068
optimizer_step: 1
candidate_check_performed: True
shared_param_paths: ['layers/streams/0/double/b', 'layers/streams/0/double/w', 'layers/streams/0/single/b', 'layers/streams/0/single/w', 'layers/streams/1/double/b', 'layers/streams/1/double/w', 'layers/streams/1/single/b', 'layers/streams/1/single/w', 'layers/streams/2/single/b', 'layers/streams/2/single/w']
grad_l2_norm: 219.0423583984375
param_delta_l2_norm: 0.0007759911823086441
optimizer_update_l2_norm: 0.0010000001639127731
share_projection_l2_norm: 0.0006307432777248323
update_accepted: True
```

```text
iteration: 1
sampler_steps: 2
sampler_acceptance: 1.0
penalty_objective: -7.228649139404297
state_energy: [-7.62111759185791, -8.419901847839355]
state_energy_std: [7.800471305847168, 0.6502408981323242]
overlap_gradient_scale: [[5.0, 5.0], [1.8639578819274902, 1.8639578819274902]]
offdiag_squared_overlap: 0.15837207436561584
optimizer_step: 2
candidate_check_performed: True
shared_param_paths: ['layers/streams/0/double/b', 'layers/streams/0/double/w', 'layers/streams/0/single/b', 'layers/streams/0/single/w', 'layers/streams/1/double/b', 'layers/streams/1/double/w', 'layers/streams/1/single/b', 'layers/streams/1/single/w', 'layers/streams/2/single/b', 'layers/streams/2/single/w']
grad_l2_norm: 186.85426330566406
param_delta_l2_norm: 0.0007543814135715365
optimizer_update_l2_norm: 0.0010000000474974513
share_projection_l2_norm: 0.0006564438808709383
update_accepted: True
```
