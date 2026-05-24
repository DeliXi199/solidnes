# Excited-State NES-VMC Tasks

This folder is the parent task area for the next SolidNES phase:

```text
Reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC method in
the SolidNES code path, then test it on concrete periodic materials.
```

Use this folder for future numbered task bundles related to:

- implementing the multi-state / penalty-based excited-state objective;
- adding overlap, orthogonality, and state-energy diagnostics;
- building controlled carbon-diamond Gamma excited-state probes;
- comparing pretraining and no-pretraining initialization for excited states;
- extending from the controlled diamond probe to specific material tests;
- recording finite-size, twist, direct-gap, and indirect-gap checks when they
  become part of the tested workflow.

Do not create a child task bundle for pure source audits, literature review, or
design notes. For example, cloning and reading `external/deepqmc/` to inspect
the Szabo-Noe penalty method should be recorded under `docs/05_reference_projects/`
or `records/progress/`, not as `0063`, unless it also produces a build, smoke,
run, evaluation, analysis, SLURM plan, log, result, or validation artifact.

Future task bundles should follow the project numbering rules:

```text
tasks/excited_state_nesvmc/NNNN_short_slug/
  results/
  outputs/
  logs/
```

Allocate `NNNN` in `records/run_index.md` before creating a child task bundle.
This parent folder itself is not a numbered task and does not consume a run
number.
