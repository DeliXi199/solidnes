# FermiNet PBC Real Local-Energy Training Smoke Failure

```text
status: failed
job_id: 128523
partition: intelgpu80g
node: gpu001
elapsed: 00:04:57
exit_code: 1:0
resources: gpu:2, cpu:96, exclusive
experiment: configs/experiment/diamond_c_ferminet_pbc_gamma_real_local_energy_training_smoke.yaml
backend_script: scripts/validation/check_ferminet_pbc_real_local_energy_training_smoke.py
```

The job started correctly in the `solidnes-ferminet-jax0101-cuda12`
environment. JAX reported the GPU backend and both A100 80GB devices:
`cuda:0` and `cuda:1`.

The failure happened during final result validation, after the tiny fixed-sample
two-step SGD loop had run through the real FermiNet PBC local-energy/Laplacian
path:

```text
ValueError: final_local_energy contains non-finite values: [[nan] [nan]]
```

Interpretation:

```text
This is not a scheduler/resource failure. Run 0064 shows that the current
direct value_and_grad path over the real local-energy objective can push the
external state parameters into non-finite local energies even with a learning
rate of 1e-8 and one walker per state. Do not resubmit the same job unchanged.
```

Next step:

```text
Implement the paper-faithful penalty-VMC training tangent before another
real-local-energy multi-step smoke. The source change should add ordered
lower-state stop-gradient overlap terms, psi-ratio and local-energy clipping,
automatic penalty scaling, and finite-gradient/update guards. This is source
work first, so it should not allocate a new numbered task folder until another
SLURM smoke or durable validation artifact is produced.
```
