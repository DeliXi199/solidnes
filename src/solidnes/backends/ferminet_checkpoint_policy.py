"""Checkpoint policy helpers for SolidNES FermiNet runs."""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


FINAL_CHECKPOINT_MIN_ITERATIONS = 1000


@dataclass(frozen=True)
class FinalCheckpointRequirement:
    """Description of the required final-step checkpoint for a run."""

    required: bool
    iterations: int
    final_step: int | None
    checkpoint_path: Path | None

    def to_json(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "iterations": self.iterations,
            "final_step": self.final_step,
            "checkpoint_path": str(self.checkpoint_path)
            if self.checkpoint_path is not None
            else None,
            "exists": self.checkpoint_path.exists()
            if self.checkpoint_path is not None
            else None,
        }


class _FinalCheckpointState:
    def __init__(self) -> None:
        self.force_checkpoint_time = False


class _FinalStepRange:
    """Iterator that marks the final training step while its body runs."""

    def __init__(
        self,
        base_range: range,
        *,
        final_step: int,
        state: _FinalCheckpointState,
    ) -> None:
        self._iterator = iter(base_range)
        self._final_step = final_step
        self._state = state

    def __iter__(self) -> "_FinalStepRange":
        return self

    def __next__(self) -> int:
        try:
            value = next(self._iterator)
        except StopIteration:
            self._state.force_checkpoint_time = False
            raise
        self._state.force_checkpoint_time = value == self._final_step
        return value


def final_checkpoint_requirement(
    cfg: Any,
    *,
    min_iterations: int = FINAL_CHECKPOINT_MIN_ITERATIONS,
) -> FinalCheckpointRequirement:
    """Return the final-step checkpoint required by the project policy."""

    iterations = int(cfg.optim.iterations)
    required = iterations >= min_iterations
    if not required:
        return FinalCheckpointRequirement(
            required=False,
            iterations=iterations,
            final_step=None,
            checkpoint_path=None,
        )
    final_step = iterations - 1
    checkpoint_path = Path(cfg.log.save_path) / f"qmcjax_ckpt_{final_step:06d}.npz"
    return FinalCheckpointRequirement(
        required=True,
        iterations=iterations,
        final_step=final_step,
        checkpoint_path=checkpoint_path,
    )


@contextmanager
def enforce_ferminet_final_checkpoint(
    train_module: ModuleType,
    cfg: Any,
    *,
    min_iterations: int = FINAL_CHECKPOINT_MIN_ITERATIONS,
) -> Iterator[FinalCheckpointRequirement]:
    """Force a time-based FermiNet checkpoint on the final training step.

    Upstream FermiNet saves checkpoints only when enough wall-clock time has
    elapsed. SolidNES long-run policy requires a checkpoint for the final
    iteration. The least invasive runtime hook is to make FermiNet's
    time-based checkpoint predicate true while the final training-step body is
    executing.
    """

    requirement = final_checkpoint_requirement(
        cfg,
        min_iterations=min_iterations,
    )
    if not requirement.required:
        yield requirement
        return

    state = _FinalCheckpointState()
    original_range = getattr(train_module, "range", None)
    had_module_range = hasattr(train_module, "range")
    original_time = train_module.time.time
    save_frequency_seconds = float(cfg.log.save_frequency) * 60.0

    def policy_range(*args: int) -> range | _FinalStepRange:
        base_range = builtins.range(*args)
        if len(base_range) and base_range[-1] == requirement.final_step:
            return _FinalStepRange(
                base_range,
                final_step=int(requirement.final_step),
                state=state,
            )
        return base_range

    def policy_time() -> float:
        now = original_time()
        if state.force_checkpoint_time:
            return now + save_frequency_seconds + 1.0
        return now

    train_module.range = policy_range
    train_module.time.time = policy_time
    try:
        yield requirement
    finally:
        train_module.time.time = original_time
        if had_module_range:
            train_module.range = original_range
        else:
            delattr(train_module, "range")


def assert_final_checkpoint_written(requirement: FinalCheckpointRequirement) -> None:
    """Raise if a policy-required final checkpoint was not produced."""

    if not requirement.required:
        return
    assert requirement.checkpoint_path is not None
    if not requirement.checkpoint_path.exists():
        raise FileNotFoundError(
            "Long SolidNES FermiNet runs must save the final-step checkpoint; "
            f"expected {requirement.checkpoint_path}"
        )
