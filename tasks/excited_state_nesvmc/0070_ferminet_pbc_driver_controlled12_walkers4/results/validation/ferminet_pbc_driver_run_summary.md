# FermiNet PBC Excited-State Driver Run

```text
status: ok
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_driver_controlled12_walkers4.yaml
jax_platform: gpu
external_state_params: 2
walkers_per_state: 4
completed_iterations: 12
checkpoint_interval: 4
learning_rate: 1e-08
local_energy_source: real_pbc
elapsed_seconds: 749.351784
initial_penalty_objective: 56.46631622314453
final_penalty_objective: -13.346861839294434
```

Checkpoints:

- `tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4/results/checkpoints/driver_iter_000004.pkl`
- `tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4/results/checkpoints/driver_iter_000008.pkl`
- `tasks/excited_state_nesvmc/0070_ferminet_pbc_driver_controlled12_walkers4/results/checkpoints/driver_iter_000012.pkl`

Per-iteration diagnostics:

```text
iteration: 0
sampler_steps: 3
sampler_acceptance: 0.7083333730697632
penalty_objective: -7.713545322418213
state_energy: [-9.348628997802734, -14.165730476379395]
offdiag_squared_overlap: 0.8087267279624939
grad_l2_norm: 9789.9287109375
param_delta_l2_norm: 9.79047326836735e-05
update_accepted: True
```

```text
iteration: 1
sampler_steps: 2
sampler_acceptance: 0.6875
penalty_objective: -16.76626968383789
state_energy: [-29.807281494140625, -12.112948417663574]
offdiag_squared_overlap: 0.8387691974639893
grad_l2_norm: 3044.3837890625
param_delta_l2_norm: 3.0464818337350152e-05
update_accepted: True
```

```text
iteration: 2
sampler_steps: 2
sampler_acceptance: 0.625
penalty_objective: -20.335237503051758
state_energy: [-30.30168914794922, -11.690871238708496]
offdiag_squared_overlap: 0.13220854103565216
grad_l2_norm: 2380.649658203125
param_delta_l2_norm: 2.3825710741220973e-05
update_accepted: True
```

```text
iteration: 3
sampler_steps: 2
sampler_acceptance: 0.75
penalty_objective: -24.433507919311523
state_energy: [-41.40237808227539, -10.317508697509766]
offdiag_squared_overlap: 0.2852869927883148
grad_l2_norm: 2002.319580078125
param_delta_l2_norm: 2.0042674805154093e-05
update_accepted: True
```

```text
iteration: 4
sampler_steps: 2
sampler_acceptance: 0.75
penalty_objective: -16.68946075439453
state_energy: [-26.469324111938477, -8.253989219665527]
offdiag_squared_overlap: 0.1344391405582428
grad_l2_norm: 1457.750732421875
param_delta_l2_norm: 1.460042767575942e-05
update_accepted: True
```

```text
iteration: 5
sampler_steps: 2
sampler_acceptance: 0.875
penalty_objective: -13.346441268920898
state_energy: [-19.390899658203125, -9.961359024047852]
offdiag_squared_overlap: 0.26593756675720215
grad_l2_norm: 496.0950012207031
param_delta_l2_norm: 4.995574272470549e-06
update_accepted: True
```

```text
iteration: 6
sampler_steps: 2
sampler_acceptance: 0.875
penalty_objective: -15.423707962036133
state_energy: [-21.702434539794922, -9.848610877990723]
offdiag_squared_overlap: 0.07036284357309341
grad_l2_norm: 623.4024658203125
param_delta_l2_norm: 6.2582321334048174e-06
update_accepted: True
```

```text
iteration: 7
sampler_steps: 2
sampler_acceptance: 0.75
penalty_objective: -15.477560043334961
state_energy: [-21.946958541870117, -10.386469841003418]
offdiag_squared_overlap: 0.13783076405525208
grad_l2_norm: 615.5570068359375
param_delta_l2_norm: 6.177789600769756e-06
update_accepted: True
```

```text
iteration: 8
sampler_steps: 2
sampler_acceptance: 0.9375
penalty_objective: -15.363819122314453
state_energy: [-23.13321876525879, -10.42460823059082]
offdiag_squared_overlap: 0.2830188572406769
grad_l2_norm: 1020.0919799804688
param_delta_l2_norm: 1.0222096534562297e-05
update_accepted: True
```

```text
iteration: 9
sampler_steps: 2
sampler_acceptance: 0.8125
penalty_objective: -13.666894912719727
state_energy: [-18.8052921295166, -10.77174186706543]
offdiag_squared_overlap: 0.22432443499565125
grad_l2_norm: 444.82904052734375
param_delta_l2_norm: 4.4848329707747325e-06
update_accepted: True
```

```text
iteration: 10
sampler_steps: 2
sampler_acceptance: 0.9375
penalty_objective: -13.1786470413208
state_energy: [-18.047958374023438, -11.904821395874023]
offdiag_squared_overlap: 0.3595486283302307
grad_l2_norm: 270.1719665527344
param_delta_l2_norm: 2.739589945122134e-06
update_accepted: True
```

```text
iteration: 11
sampler_steps: 2
sampler_acceptance: 1.0
penalty_objective: -13.437276840209961
state_energy: [-17.740276336669922, -11.480193138122559]
offdiag_squared_overlap: 0.23459166288375854
grad_l2_norm: 337.12249755859375
param_delta_l2_norm: 3.411984607737395e-06
update_accepted: True
```
