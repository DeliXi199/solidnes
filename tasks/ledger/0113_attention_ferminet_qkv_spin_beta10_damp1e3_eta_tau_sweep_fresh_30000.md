# Task 0113: FermiNet-QKV Eta/Tau Fresh 30000 Sweep

Task root:

```text
tasks/psiformer/0113_attention_ferminet_qkv_spin_beta10_damp1e3_eta_tau_sweep_fresh_30000
```

This task runs six fresh 30000-step two-state PsiFormer excited-state jobs.
All variants use the FermiNet-QKV attention route, deterministic initialization,
DeepQMC-style state-specific spin penalty with `beta=10.0`, KFAC
`damping=0.001`, and `norm_constraint=0.001`. The only swept parameters are
`eta0` and `tau` in `eta(t)=eta0/(1+t/tau)`.

The jobs are fresh starts: no checkpoint restore path is configured. Normal
completion is expected to write `qmcjax_ckpt_029999.npz`, and the FermiNet
runner enforces final-checkpoint creation on clean completion.

Submitted jobs:

| Job ID | eta0 | tau | Latest status |
| ---: | ---: | ---: | --- |
| 139207 | `0.01` | `15000` | R on `gpuh2001` |
| 139214 | `0.01` | `20000` | R on `gpu004` |
| 139209 | `0.01` | `10000` | R on `gpu006` |
| 139215 | `0.02` | `15000` | PD on `h200,amdgpu80g,amdgpu40g` |
| 139216 | `0.02` | `20000` | PD on `h200,amdgpu80g,amdgpu40g` |
| 139217 | `0.02` | `10000` | PD on `h200,amdgpu80g,amdgpu40g` |

Slurm time limit is 24 hours for every job. Plans and logs are written under
the task root.

The original pending jobs `139208`, `139210`, `139211`, and `139212` were
cancelled before running and replaced with the flexible partition-queue
submissions above. The replacement `sbatch` commands use
`--partition h200,amdgpu80g,amdgpu40g` without `--nodelist`.
