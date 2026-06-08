# Task 0114: FermiNet-QKV Low-Eta/Tau Fresh 30000 Sweep

Task root:

```text
tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000
```

This task follows the completed 0113 fresh-start eta/tau sweep but lowers the
base learning rate. All six variants use FermiNet-QKV PsiFormer, deterministic
fresh initialization, `beta=10.0`, KFAC `damping=0.001`, no parameter merging,
and 30000 iterations.

Swept parameters:

| eta0 | tau |
| ---: | ---: |
| `0.005` | `10000` |
| `0.005` | `15000` |
| `0.005` | `20000` |
| `0.001` | `10000` |
| `0.001` | `15000` |
| `0.001` | `20000` |

Submitted jobs:

| Job ID | eta0 | tau | Latest status |
| ---: | ---: | ---: | --- |
| 139618 | `0.005` | `10000` | COMPLETED `0:0` on `gpuh2001` |
| 139657 | `0.005` | `15000` | COMPLETED `0:0` on `gpu004`; original 139619 cancelled |
| 139658 | `0.005` | `20000` | COMPLETED `0:0` on `gpu002`; original 139620 cancelled |
| 139659 | `0.001` | `10000` | COMPLETED `0:0` on `gpuh2001`; original 139621 cancelled |
| 139660 | `0.001` | `15000` | COMPLETED `0:0` on `gpuh2001`; original 139622 cancelled |
| 139661 | `0.001` | `20000` | COMPLETED `0:0` on `gpuh2001`; original 139623 cancelled |

Every job uses flexible Slurm queueing over `h200,amdgpu80g,amdgpu40g` with
4 GPUs, 64 CPUs, 64 GB memory, and a 24-hour time limit. Plans and logs are
written under the task root.

Jobs 139619-139623 were cancelled while still pending, then resubmitted as
139657-139661. Jobs 139618 and 139657-139661 completed with exit code `0:0`.
Analysis outputs are in
`tasks/psiformer/0114_attention_ferminet_qkv_spin_beta10_damp1e3_low_eta_tau_sweep_fresh_30000/analysis/`.

The combined 0113/0114 fixed-tau eta analysis now also includes 1000-step
rolling mean, rolling standard deviation, rolling variance, mean absolute
one-step-change, and EW variance plots under
`analysis/fixed_tau_eta_comparison/`.

Default parameter decision: use `eta0=0.02`, `tau=10000`, and
`learning_rate_decay=1.0` for future PsiFormer excited-state calculations unless
a task explicitly declares a new sweep or ablation. The decision record and a
fixed tarball snapshot of the generated analysis data are saved under
`records/analysis/2026-06-08_excited_state_default_eta2e2_tau10000/`.
