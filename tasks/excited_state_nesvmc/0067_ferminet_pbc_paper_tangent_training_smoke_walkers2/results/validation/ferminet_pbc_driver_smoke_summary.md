# FermiNet PBC Driver Smoke

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml
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
kfac_momentum: None
kfac_norm_constraint: 0.001
kfac_invert_every: 1
overlap_ewma_decay: 0.5
param_share_keys: ['layers/streams']
candidate_check_period: 1
gradient_mode: paper_tangent
local_energy_source: cheap
checkpoint_roundtrip_bytes: 3702890
initial_penalty_objective: 13.869477272033691
final_penalty_objective: 13.884084701538086
```

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 0.9166666865348816
penalty_objective: 13.884084701538086
gradient_objective: 13.884084701538086
state_energy: [13.422319412231445, 14.34584903717041]
offdiag_squared_overlap: 0.0
optimizer_step: 1
candidate_check_performed: True
shared_param_paths: ['layers/streams/0/double/b', 'layers/streams/0/double/w', 'layers/streams/0/single/b', 'layers/streams/0/single/w', 'layers/streams/1/double/b', 'layers/streams/1/double/w', 'layers/streams/1/single/b', 'layers/streams/1/single/w', 'layers/streams/2/single/b', 'layers/streams/2/single/w']
grad_l2_norm: 19.584205627441406
param_delta_l2_norm: 0.00078001240035519
optimizer_update_l2_norm: 0.0010000000474974513
share_projection_l2_norm: 0.00062576710479334
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
```
