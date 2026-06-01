# Research Workflow

## Default Session Flow

1. Read `AGENTS.md`.
2. Read `CURRENT_CONTEXT.md`.
3. Read `ACTIVE_TASK.md` only when the user asks about the current small step
   or the work will change task state.
4. Search `records/progress/` or task README files only when historical detail
   is needed.
5. Make one small, reviewable change.
6. Record what changed and what remains uncertain.

## What To Record

Record decisions when they affect:

- The target system.
- The ansatz.
- The optimizer.
- The excited-state objective.
- The compute budget.
- The interpretation of a result.

Record progress when:

- A file or folder is added.
- A paper is summarized.
- An implementation path is chosen.
- An experiment is started, finished, or abandoned.
- A result changes the project direction.

## File Conventions

Progress notes:

```text
records/progress/YYYY-MM-DD_short_title.md
```

Decision records:

```text
records/decisions/0001_short_title.md
```

Experiment folders:

```text
experiments/phaseN_system_target/
```
