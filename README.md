# SolidNES

Neural Quantum States for Excited States of Periodic Solids.

SolidNES is a research project for testing whether excited-state neural quantum
state methods, especially NES-VMC, can be combined with periodic solid-state
wavefunction ansatzes to compute excited states of real materials from first
principles.

## Current Focus

The project has completed the carbon-diamond ground-state benchmark
reproduction milestone through both DeepSolid and FermiNet. The current route
is to continue method integration on carbon diamond, with FermiNet as the
primary efficiency and future-development framework.

Passed so far:

- DeepSolid import/config smoke in an isolated JAX 0.4.30 probe environment.
- Carbon diamond DeepSolid config construction.
- Zero-iteration carbon diamond DeepSolid runtime smoke on CPU and CUDA GPU.
- Decision to continue with DeepSolid as the first Phase 1 backend.
- Reusable DeepSolid adapter module.
- One-iteration carbon diamond Adam smoke on CUDA GPU.
- Direct DeepSolid adapter-object probe on CPU and CUDA GPU: initialized
  network apply functions, params, walkers, local energy, and MCMC transition
  outside DeepSolid's training loop.
- Carbon-diamond validation harness: PySCF PBC HF reference plus DeepSolid
  training stats summaries. The harness works, but current short `sto-3g`
  validation runs are not yet accurate or converged.
- FermiNet x64 carbon-diamond paper-geometry benchmark and fixed-checkpoint
  evaluation.

Phase 1 remains deliberately narrow:

- Build a periodic excited-state NQS workflow.
- Use carbon diamond as the controlled periodic-solid benchmark while the
  backend path is hardened.
- Add excited-state/NES-VMC machinery only after the ground-state route is
  reproducible and well-instrumented.

## Start Here

- [AGENTS.md](AGENTS.md): global operating instructions for assistants and
  coding agents; read this first before working in the project.
- [CURRENT_STATUS.md](CURRENT_STATUS.md): current milestone, evidence jobs,
  caveats, and next phase.
- [ACTIVE_TASK.md](ACTIVE_TASK.md): exact current small step, job state, next
  action, expected output, and completion condition.
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): project scope, rules, and working style.
- [PROGRESS.md](PROGRESS.md): short rolling current-state snapshot.
- [ROADMAP.md](ROADMAP.md): phase plan and milestones.
- [DIRECTORY_MAP.md](DIRECTORY_MAP.md): what belongs in each folder.
- [docs/00_project_guidance/implementation_strategy.md](docs/00_project_guidance/implementation_strategy.md): first implementation strategy.
- [docs/00_project_guidance/backend_survey.md](docs/00_project_guidance/backend_survey.md): backend comparison and first backend choice.
- [docs/00_project_guidance/deepsolid_environment_plan.md](docs/00_project_guidance/deepsolid_environment_plan.md): DeepSolid environment and smoke-test status.
- [docs/00_project_guidance/deepsolid_adapter_interface.md](docs/00_project_guidance/deepsolid_adapter_interface.md): current backend adapter boundary.
- [docs/00_project_guidance/slurm_task_management.md](docs/00_project_guidance/slurm_task_management.md): CPU/GPU SLURM task management.
- [docs/00_project_guidance/result_output_numbering.md](docs/00_project_guidance/result_output_numbering.md): numbered task-bundle rules.
- [records/run_index.md](records/run_index.md): canonical ledger for numbered result runs.
- [tasks/TASK_LEDGER.md](tasks/TASK_LEDGER.md): readable ledger of numbered
  task purpose and key results.

## Repository Layout

```text
solidnes/
  AGENTS.md             # Global instructions for assistants/coding agents
  CURRENT_STATUS.md     # Current project-level state and milestone summary
  ACTIVE_TASK.md        # Current small-step task state and next action
  PROJECT_GUIDE.md       # Project principles and operating rules
  PROGRESS.md            # Short rolling current-state snapshot
  ROADMAP.md             # Phase-level research plan
  DIRECTORY_MAP.md       # Folder rules and structure
  docs/                  # Guidance, theory notes, literature, reports
  records/               # Progress logs and decision records
  records/run_index.md   # Numbered task-bundle ledger
  configs/               # Reproducible experiment configs
  experiments/           # Phase-specific run and analysis folders
  tasks/                 # Numbered task bundles: results, outputs, logs together
  tasks/ferminet_pretraining/
                         # FermiNet PBC-HF pretraining task bundles
  tasks/TASK_LEDGER.md   # Readable ledger of numbered task purposes and results
  src/solidnes/          # SolidNES package source and backend adapters
  references/            # Local reference notes and citation material
  external/              # Ignored local backend checkouts/copies
  patches/               # Patch records for external backend changes
  scripts/               # Utility scripts and cluster launchers
  .venv/                 # Ignored local Python environment
```
