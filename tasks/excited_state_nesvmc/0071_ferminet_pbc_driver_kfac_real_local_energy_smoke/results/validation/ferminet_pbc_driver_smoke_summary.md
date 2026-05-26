# FermiNet PBC Driver Smoke

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_driver_kfac_real_local_energy_smoke.yaml
jax_platform: cpu
external_state_params: 2
walkers_per_state: 2
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
candidate_check_period: 1
gradient_mode: paper_tangent
local_energy_source: cheap
checkpoint_roundtrip_bytes: 3702888
initial_penalty_objective: 13.684469223022461
final_penalty_objective: 13.642754554748535
```

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 1.0
penalty_objective: 13.641745567321777
gradient_objective: 13.641745567321777
state_energy: [13.36059856414795, 13.685006141662598]
offdiag_squared_overlap: 0.18360507488250732
optimizer_step: 1
candidate_check_performed: True
shared_param_paths: ['layers/streams/0/double/b', 'layers/streams/0/double/w', 'layers/streams/0/single/b', 'layers/streams/0/single/w', 'layers/streams/1/double/b', 'layers/streams/1/double/w', 'layers/streams/1/single/b', 'layers/streams/1/single/w', 'layers/streams/2/single/b', 'layers/streams/2/single/w']
grad_l2_norm: 19.036123275756836
param_delta_l2_norm: 0.0007782859611324966
optimizer_update_l2_norm: 0.0010000001639127731
share_projection_l2_norm: 0.0006279131048358977
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
```
