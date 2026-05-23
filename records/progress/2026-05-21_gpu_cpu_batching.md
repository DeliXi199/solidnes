# Progress: GPU CPU Batching

Date: 2026-05-21

## Summary

SolidNES updated the GPU SLURM launch path so CPU cores allocated with a GPU job
are no longer ignored by default.

The scheduler now exports CPU-to-GPU batch metadata, and the GPU run script uses
the allocated CPU budget to set host thread pools.

## Files

```text
src/solidnes/slurm_scheduling.py
scripts/slurm/run_deepsolid_gpu_smoke.slurm
```

## Behavior

For a GPU job with 2 GPUs and 16 CPUs, the dry-run now exports:

```text
SOLIDNES_TOTAL_GPUS=2
SOLIDNES_TOTAL_CPU_CORES=16
SOLIDNES_CPUS_PER_GPU_BASE=8
SOLIDNES_CPU_REMAINDER=0
SOLIDNES_CPUS_PER_GPU_LIST=8:8
SOLIDNES_CPU_BATCHES=gpu0:0-7;gpu1:8-15
SOLIDNES_CPU_PARALLELISM=16
```

If an exclusive A100 node gives 96 CPUs and 2 GPUs, the same logic produces
48 CPU slots per GPU:

```text
SOLIDNES_CPUS_PER_GPU_LIST=48:48
SOLIDNES_CPU_BATCHES=gpu0:0-47;gpu1:48-95
SOLIDNES_CPU_PARALLELISM=96
```

The runtime script now logs these values and defaults the CPU thread pools to
the allocated CPU parallelism:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

The default `SOLIDNES_CPU_THREAD_MODE=all` gives the single DeepSolid/JAX host
process access to the full allocated CPU pool. `SOLIDNES_CPU_THREAD_MODE=per_gpu`
can be used to cap each CPU thread pool at the per-GPU base count.

## Validation

Checks passed:

```text
python -m py_compile src/solidnes/slurm_scheduling.py scripts/slurm/plan_slurm_job.py
bash -n scripts/slurm/run_deepsolid_gpu_smoke.slurm scripts/slurm/submit_deepsolid_gpu_smoke.sh
SLURM dry-run for diamond_c_deepsolid_validation_iter5000_batch96_mcmc12_full
```

The dry-run produced the expected CPU batching export in the `sbatch` command.
