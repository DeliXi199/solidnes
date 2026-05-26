# FermiNet PBC Driver Smoke

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_iter100_fullnode.yaml
jax_platform: cpu
external_state_params: 2
walkers_per_state: 4
iterations: 1
burn_in: 1
sampler_steps_per_iteration: 2
proposal_width: 0.02
optimizer: kfac
learning_rate: 0.0001
kfac_damping: 0.001
kfac_momentum: 0.0
kfac_norm_constraint: 0.001
kfac_invert_every: 1
overlap_ewma_decay: 0.5
param_share_keys: ['layers/streams']
candidate_check_period: 5
gradient_mode: paper_tangent
local_energy_source: cheap
checkpoint_roundtrip_bytes: 6166552
initial_penalty_objective: 15.086295127868652
final_penalty_objective: 12.735107421875
```

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 0.75
penalty_objective: 12.735107421875
gradient_objective: 12.735107421875
state_energy: [13.122798919677734, 12.34741497039795]
offdiag_squared_overlap: 0.0
optimizer_step: 1
candidate_check_performed: True
shared_param_paths: ['layers/streams/0/double/b', 'layers/streams/0/double/w', 'layers/streams/0/single/b', 'layers/streams/0/single/w', 'layers/streams/1/double/b', 'layers/streams/1/double/w', 'layers/streams/1/single/b', 'layers/streams/1/single/w', 'layers/streams/2/single/b', 'layers/streams/2/single/w']
grad_l2_norm: 0.0010000001639127731
param_delta_l2_norm: 0.0007966572884470224
optimizer_update_l2_norm: 0.0010000001639127731
share_projection_l2_norm: 0.0006044295732863247
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
```
