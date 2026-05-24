# 2026-05-24 Excited-State Next Phase Scaffold

## Update

Recorded the next SolidNES phase as:

```text
Reproduce the Szabo and Noe JCTC 2024 penalty-based excited-state VMC method in
the SolidNES code path, then test it on concrete periodic materials.
```

## Files Updated

- `CURRENT_STATUS.md`
- `ACTIVE_TASK.md`
- `PROGRESS.md`
- `ROADMAP.md`
- `PROJECT_GUIDE.md`
- `AGENTS.md`
- `README.md`
- `DIRECTORY_MAP.md`

## New Task Area

Created:

```text
tasks/excited_state_nesvmc/
```

This folder is the parent area for future numbered task bundles related to:

- implementing the paper-style penalty-based excited-state VMC objective;
- adding state-energy, overlap, and orthogonality diagnostics;
- running the first controlled carbon-diamond Gamma excited-state probes;
- testing the resulting workflow on specific materials;
- recording direct-gap, indirect-gap, twist, and finite-size caveats.

The parent folder does not consume a run number. Future child task bundles
should use the next available run number from `records/run_index.md`.
