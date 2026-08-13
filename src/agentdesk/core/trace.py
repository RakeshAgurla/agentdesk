"""Run tracing.

When a multi-agent run produces a bad answer, the question is always "which
step went wrong". Without a trace the only available answer is to re-run it and
watch, which does not work when the failure is intermittent.

Every step records what went in, what came out, what it cost, and whether it
succeeded. The trace is serialisable so a failing run can be attached to a bug
report and replayed.

Deliberately not a logging wrapper. Logs are for humans reading in real time;
this is a structured record the system itself reasons about -- the orchestrator
inspects prior attempts to decide whether a retry is worth making.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self


class StepStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"       # ran fine, output failed validation
    ERROR = "error"             # the step itself blew up
    BUDGET_EXCEEDED = "budget_exceeded"
    SKIPPED = "skipped"


@dataclass
class StepTrace:
    step_id: str
    agent: str
    status: StepStatus
    started_at: float
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attempt: int = 1
    summary: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    outcome: str = "incomplete"

    def add(self, step: StepTrace) -> None:
        self.steps.append(step)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def duration(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def attempts_for(self, agent: str) -> int:
        return sum(1 for s in self.steps if s.agent == agent)

    def rejections(self) -> list[StepTrace]:
        return [s for s in self.steps if s.status is StepStatus.REJECTED]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "outcome": self.outcome,
            "duration_s": round(self.duration, 3),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def render(self) -> str:
        """Human-readable summary. What you actually look at first."""
        icons = {
            StepStatus.OK: "ok  ",
            StepStatus.REJECTED: "rej ",
            StepStatus.ERROR: "err ",
            StepStatus.BUDGET_EXCEEDED: "budg",
            StepStatus.SKIPPED: "skip",
        }
        lines = [
            f"run {self.run_id}  outcome={self.outcome}",
            (
                f"{len(self.steps)} steps | {self.total_tokens} tokens | "
                f"${self.total_cost:.4f} | {self.duration:.2f}s"
            ),
            "",
        ]
        for i, s in enumerate(self.steps, 1):
            lines.append(
                f"  {i:>2}. [{icons[s.status]}] {s.agent:<12} "
                f"attempt {s.attempt}  {s.duration_s:>5.2f}s  "
                f"{s.input_tokens + s.output_tokens:>5}tok  {s.summary}"
            )
            if s.error:
                lines.append(f"        error: {s.error}")
        return "\n".join(lines)


class Tracer:
    """Context manager that times a step and records it either way."""

    def __init__(self, run: RunTrace, agent: str, attempt: int = 1):
        self.run = run
        self.agent = agent
        self.attempt = attempt
        self.step: StepTrace | None = None
        self._start = 0.0

    def __enter__(self) -> Self:
        self._start = time.monotonic()
        self.step = StepTrace(
            step_id=uuid.uuid4().hex[:8],
            agent=self.agent,
            status=StepStatus.OK,
            started_at=time.time(),
            duration_s=0.0,
            attempt=self.attempt,
        )
        return self

    def record_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        assert self.step is not None
        self.step.input_tokens = input_tokens
        self.step.output_tokens = output_tokens
        self.step.cost_usd = cost

    def set(self, status: StepStatus, summary: str = "", **detail) -> None:
        assert self.step is not None
        self.step.status = status
        if summary:
            self.step.summary = summary
        self.step.detail.update(detail)

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.step is not None
        self.step.duration_s = round(time.monotonic() - self._start, 4)
        if exc is not None:
            self.step.status = StepStatus.ERROR
            self.step.error = f"{exc_type.__name__}: {exc}"
        self.run.add(self.step)
        return False  # never swallow; the orchestrator decides what is fatal
