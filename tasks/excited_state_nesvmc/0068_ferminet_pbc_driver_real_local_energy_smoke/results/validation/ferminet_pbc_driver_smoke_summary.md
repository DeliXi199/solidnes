# FermiNet PBC Driver Smoke

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_driver_real_local_energy_smoke.yaml
jax_platform: gpu
external_state_params: 2
walkers_per_state: 2
iterations: 2
burn_in: 1
sampler_steps_per_iteration: 2
proposal_width: 0.02
learning_rate: 1e-08
gradient_mode: paper_tangent
local_energy_source: real_pbc
checkpoint_roundtrip_bytes: 1233826
initial_penalty_objective: -1.3069219589233398
final_penalty_objective: 21.307846069335938
```

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 0.75
penalty_objective: 19.08087730407715
gradient_objective: 19.08087730407715
state_energy: [2.3737258911132812, -1.5953292846679688]
offdiag_squared_overlap: 3.7383358478546143
grad_l2_norm: 5728.587890625
param_delta_l2_norm: 5.729706754209474e-05
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
```

```text
iteration: 1
sampler_steps: 2
sampler_acceptance: 0.75
penalty_objective: 17.955150604248047
gradient_objective: 17.955150604248047
state_energy: [1.6127510070800781, -2.0784244537353516]
offdiag_squared_overlap: 4.3110432624816895
grad_l2_norm: 6774.43408203125
param_delta_l2_norm: 6.774779467377812e-05
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
```
