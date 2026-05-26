# FermiNet PBC Real Local-Energy Training Smoke

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_paper_tangent_training_smoke_walkers2.yaml
jax_platform: gpu
cfg_system_states: 0
external_state_params: 2
walkers_per_state: 2
steps: 2
learning_rate: 1e-08
gradient_mode: paper_tangent
local_energy_source: real_pbc
elapsed_seconds: 279.276906
initial_penalty_objective: -11.952571868896484
final_penalty_objective: -11.967228889465332
objective_delta: -0.014657020568847656
initial_state_energy: [-10.147720336914062, -14.60999870300293]
final_state_energy: [-10.17345142364502, -14.60450553894043]
```

Per-step diagnostics:

```text
step: 0
penalty_objective: -11.952571868896484
gradient_objective: -11.952571868896484
state_energy: [-10.147720336914062, -14.60999870300293]
overlap_penalty: 0.08525747060775757
overlap_gradient_scale: [[5.0, 5.0], [4.462278366088867, 2.6380672454833984]]
grad_l2_norm: 557.01806640625
param_delta_l2_norm: 5.602138116955757e-06
gradient_objective_finite: True
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
collapse_flag: False
```

```text
step: 1
penalty_objective: -12.040141105651855
gradient_objective: -12.040141105651855
state_energy: [-10.197734832763672, -14.740167617797852]
overlap_penalty: 0.08576193451881409
overlap_gradient_scale: [[5.0, 5.0], [4.54243278503418, 2.763833999633789]]
grad_l2_norm: 555.3357543945312
param_delta_l2_norm: 5.5834761951700784e-06
gradient_objective_finite: True
gradient_finite: True
update_finite: True
candidate_terms_finite: True
update_accepted: True
collapse_flag: False
```

This smoke runs the reusable fixed-sample external-state penalty
training loop with the real FermiNet PBC local-energy/Laplacian
path. It checks finite diagnostics and nonzero parameter updates,
not variational convergence.
