# Implementation Strategy Scaffold

Date: 2026-05-21

## Summary

Added the first implementation strategy for SolidNES and converted the Phase 1
config plan from a single placeholder YAML into a role-split config scaffold.

## Added

- `docs/00_project_guidance/implementation_strategy.md`
- `docs/05_reference_projects/vmc_reproduce_notes.md`
- `records/decisions/0002_initial_implementation_strategy.md`
- Role-split config directories:
  - `configs/experiment/`
  - `configs/system/`
  - `configs/model/`
  - `configs/sampler/`
  - `configs/train/`

## Current Direction

The project will use an adapter-first strategy until the backend survey is
complete.

## Next Step

Audit candidate backends and determine the ground-state periodic-solid
smoke-test route.
