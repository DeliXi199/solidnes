# Scripts

Small command-line utilities and cluster launchers.

## Subfolders

- `backends/`: backend checks and smoke runners.
- `slurm/`: FIIR-style SLURM planners, submit wrappers, and job templates.
- `maintenance/`: repository and task-layout checks.

Run the task-bundle structure check with:

```bash
python scripts/maintenance/validate_task_bundles.py
```

Utility scripts and cluster launchers will live here.

Keep scripts small and tied to reproducible configs.

Backend-specific notes and wrappers live in `scripts/backends/`.
