# Decision 0008: Use FIIR-Style SLURM Planning

Date: 2026-05-21

## Decision

SolidNES will use a FIIR-style SLURM scheduling layer:

```text
standard-library planner -> deterministic plan JSON -> optional sbatch submit
```

## Rationale

CPU and GPU jobs will become expensive quickly. Hand-written `sbatch` commands
make it too easy to lose track of partition choice, resource shape, and runtime
environment. The FIIR pattern already solves this by separating scheduling
policy from job templates.

## Consequences

- Scheduling policy lives in `src/solidnes/slurm_scheduling.py`.
- Submit wrappers live in `scripts/slurm/`.
- New job shapes must support dry-run planning.
- SLURM logs go to `logs/slurm/`.
- Plan JSON files go to `outputs/slurm_plans/`.

## Revisit Trigger

Revisit after the first real GPU training smoke, especially if DeepSolid needs
full-node GPU allocations or a cluster-specific partition policy.
