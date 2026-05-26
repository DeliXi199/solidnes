# 2026-05-25: Submitted Native FermiNet Spin Beta Sweep, 2000 Iterations

## Summary

Created and submitted task `0086`, a grouped 12-point spin-penalty beta sweep
for the paper-aligned native FermiNet PBC two-state `vmc_overlap` workflow.

Task root:

```text
tasks/excited_state_nesvmc/0086_ferminet_native_vmc_overlap_kfac_paper_aligned_spin_beta_sweep_iter2000
```

## Sweep Parameters

```text
spin beta: 0.000, 0.001, 0.002, 0.005, 0.008, 0.010,
           0.012, 0.015, 0.018, 0.020, 0.025, 0.030
iterations: 2000
batch size: 4096
optimizer: KFAC
overlap penalty alpha: 4.0
observables_s2: true
bare_energy_matrix output: true
partition: amdgpu80g
resources: one full node, 4 GPUs, 64 CPU cores, exclusive allocation,
           full node memory
```

## Jobs

| Variant | Beta | Job | State at submission check |
| --- | ---: | ---: | --- |
| `beta0000` | 0.000 | 129314 | RUNNING on `gpu002` |
| `beta0001` | 0.001 | 129327 | PENDING |
| `beta0002` | 0.002 | 129328 | PENDING |
| `beta0005` | 0.005 | 129329 | PENDING |
| `beta0008` | 0.008 | 129330 | PENDING |
| `beta0010` | 0.010 | 129331 | PENDING |
| `beta0012` | 0.012 | 129332 | PENDING |
| `beta0015` | 0.015 | 129333 | PENDING |
| `beta0018` | 0.018 | 129334 | PENDING |
| `beta0020` | 0.020 | 129335 | PENDING |
| `beta0025` | 0.025 | 129336 | PENDING |
| `beta0030` | 0.030 | 129337 | PENDING |

## Resource Correction

The initial pending submissions 129315--129325 were cancelled before running
because the flexible queue path inherited the default `--mem 64000` request.
The final pending jobs 129327--129337 were re-submitted with
`SOLIDNES_GPU_QUEUE_MEMORY_MB=0`; Slurm now reports the same full-node request
style as the running job:

```text
ReqTRES=cpu=64,mem=500G,node=1,billing=64,gres/gpu=4
MinMemoryNode=0
```

## Follow-up Queue Check

After re-submission, job 129314 (`beta0000`) completed in `00:06:42`; job
129327 (`beta0001`) was running on `gpu002`; jobs 129328--129337 remained
pending. The checked running and pending jobs both reported:

```text
ReqTRES=cpu=64,mem=500G,node=1,billing=64,gres/gpu=4
MinMemoryNode=0
```
