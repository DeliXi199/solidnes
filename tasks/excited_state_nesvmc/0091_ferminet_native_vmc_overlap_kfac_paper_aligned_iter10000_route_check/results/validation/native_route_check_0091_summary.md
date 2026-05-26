# 0091 Native Route Check Summary

## Status

- Slurm job: `130365`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:20:33`
- Node: `amdgpu40g/gpu007`
- Rows written: `10000`

## Numerical Health

- Last finite step: `3814`
- First non-finite step: `3815`
- Finite rows: `3815`
- Non-finite rows: `6185`
- Last finite row:

```text
step=3814, energy=-74.71772 Ha, ewmean=-74.784744 Ha,
ewvar=0.08745663 Ha^2, pmove=0.5415039
```

The job completed and wrote the expected diagnostic arrays, but the trajectory
became non-finite after step `3814`.  This proves the original native
`vmc_overlap` route still launches and executes under the current code, but it
does not prove the route is numerically stable for this run.
