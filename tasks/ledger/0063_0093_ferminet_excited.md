# Tasks 0063-0093: FermiNet Excited-State Route

This slice covers the native FermiNet PBC excited-state implementation,
spin-penalty experiments, fixed-ground attempts, and route checks before the
PsiFormer/DeepQMC alignment sequence.

Canonical detailed history:

```text
records/archive/2026-06-01_context_split/TASK_LEDGER.md
records/run_index.md
tasks/excited_state_nesvmc/**/README.md
records/progress/
```

Key milestones:

| Range | Summary |
| --- | --- |
| 0063-0070 | Real PBC local-energy/Laplacian smokes, fixed-sample training helpers, sampler-integrated driver, and first controlled short trajectories. |
| 0071-0079 | KFAC/native `vmc_overlap` route hardening, overlap diagnostics, and a stable 10000-step beta=0/no-spin native route check. |
| 0080-0088 | Spin-penalty plumbing, bare-energy reporting, beta sweeps, beta=10 pressure test, and 100000-step beta=0 baseline. |
| 0089-0090 | Fixed-ground overlap attempts; not production-ready because gap/penalty behavior was unreliable. |
| 0091-0093 | Native-route reruns and fast NaN guard; `0093` completed finite but gap remained noisy. |
