# Backend Survey

Date: 2026-05-21

## Summary

Completed the first backend survey for SolidNES.

Inspected:

- `google-deepmind/ferminet`
- `bytedance/DeepSolid`
- `bytedance/netobs`

## Main Finding

DeepSolid is the best first target for the periodic ground-state smoke path.
FermiNet is the best reference for NES-VMC and excited-state machinery. NetObs
is useful as an adapter/interface reference.

## Files Added

- `docs/00_project_guidance/backend_survey.md`
- `configs/model/deepsolid_detnet_small.yaml`
- `scripts/backends/deepsolid_run_config.md`

## Next Step

Confirm the DeepSolid runtime environment and run the carbon-diamond smoke
test.
