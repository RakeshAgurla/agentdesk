"""Orchestration.

An explicit state machine: named nodes, and a transition function that maps
(node, result) to the next node. No framework.

Why not LangGraph. It is a reasonable choice and at larger scale it earns its
keep -- persistence, streaming, human-in-the-loop checkpoints. At this size it
would hide the only part worth showing. The control flow *is* the engineering
here: what happens when validation fails twice, what happens when the budget
runs out mid-retry, what the system returns when it cannot finish. Wrapping
that in a framework would mean the interesting decisions live in someone else's
library rather than in this file.

The trade-off is real and goes in the README: this does not persist state
across process restarts, and adding that would be the point to reach for a
framework rather than reimplement it.

Terminal states are deliberately distinct. "Finished with an approved draft",
"finished with a draft that never passed validation", and "stopped because the
budget ran out" are three different outcomes and a caller needs to tell them
apart. Collapsing them into a bare string would be the easy mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agentdesk.agents.roles import (
    Document,
    DrafterAgent,
    PlannerAgent,
    RetrieverAgent,
    RunState,
    ValidatorAgent,
)
from agentdesk.core.budget import Budget, BudgetExceeded
from agentdesk.core.trace import RunTrace, StepStatus, Tracer
from agentdesk.llm.client import LLMClient, build_client


class Node(str, Enum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    DRAFT = "draft"
    VALIDATE = "validate"
    DONE = "done"
    FAILED = "failed"


class Outcome(str, Enum):
    APPROVED = "approved"
    UNVALIDATED = "unvalidated"        # draft exists, never passed validation
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"                  # could not produce a draft at all


@dataclass
class RunResult:
    outcome: Outcome
    answer: str
    trace: RunTrace
    budget: dict
    validation_notes: list[str]

    @property
    def trustworthy(self) -> bool:
        """Only an approved draft should be shown to a user unqualified."""
        return self.outcome is Outcome.APPROVED


class Orchestrator:
    def __init__(
        self,
        client: LLMClient | None = None,
        budget: Budget | None = None,
        max_draft_attempts: int = 3,
        min_headroom_for_retry: float = 0.15,
    ):
        self.client = client or build_client()
        self.budget = budget or Budget()
        self.max_draft_attempts = max_draft_attempts
        # Do not start a retry there is not enough budget left to finish. A
        # retry that dies halfway costs the same as one that completes and
        # produces nothing.
        self.min_headroom_for_retry = min_headroom_for_retry

        self.planner = PlannerAgent(self.client, self.budget)
        self.retriever = RetrieverAgent(self.client, self.budget)
        self.drafter = DrafterAgent(self.client, self.budget)
        self.validator = ValidatorAgent(self.client, self.budget)

    def run(self, task: str, corpus: list[Document]) -> RunResult:
        state = RunState(task=task, corpus=corpus)
        trace = RunTrace(task=task)
        node = Node.PLAN
        outcome = Outcome.FAILED

        while node not in (Node.DONE, Node.FAILED):
            try:
                node, outcome = self._step(node, state, trace)
            except BudgetExceeded as exc:
                with Tracer(trace, "orchestrator") as t:
                    t.set(
                        StepStatus.BUDGET_EXCEEDED,
                        f"{exc.limit_name} limit reached",
                        limit=exc.limit_name,
                    )
                outcome = (
                    Outcome.UNVALIDATED if state.draft else Outcome.BUDGET_EXHAUSTED
                )
                break

        trace.outcome = outcome.value
        trace.finished_at = __import__("time").time()

        answer = state.final or state.draft
        return RunResult(
            outcome=outcome,
            answer=answer,
            trace=trace,
            budget=self.budget.snapshot(),
            validation_notes=state.validation_notes,
        )

    def _step(
        self, node: Node, state: RunState, trace: RunTrace
    ) -> tuple[Node, Outcome]:
        if node is Node.PLAN:
            with Tracer(trace, "planner") as t:
                result = self.planner.run(state)
                t.record_usage(result.input_tokens, result.output_tokens, result.cost_usd)
                if not result.ok:
                    t.set(StepStatus.ERROR, result.reason)
                    return Node.FAILED, Outcome.FAILED
                state.plan = result.data["plan"]
                t.set(StepStatus.OK, f"{len(state.plan)} steps")
            return Node.RETRIEVE, Outcome.FAILED

        if node is Node.RETRIEVE:
            with Tracer(trace, "retriever") as t:
                result = self.retriever.run(state)
                if not result.ok:
                    t.set(StepStatus.ERROR, result.reason)
                    return Node.FAILED, Outcome.FAILED
                state.retrieved = result.data["retrieved"]
                t.set(
                    StepStatus.OK,
                    f"{len(state.retrieved)} docs",
                    top_score=result.data.get("top_score"),
                )
            return Node.DRAFT, Outcome.FAILED

        if node is Node.DRAFT:
            state.draft_attempts += 1
            with Tracer(trace, "drafter", attempt=state.draft_attempts) as t:
                result = self.drafter.run(state)
                t.record_usage(result.input_tokens, result.output_tokens, result.cost_usd)
                if not result.ok:
                    t.set(StepStatus.ERROR, result.reason)
                    return Node.FAILED, Outcome.FAILED
                state.draft = result.data["draft"]
                t.set(StepStatus.OK, f"{len(state.draft)} chars")
            return Node.VALIDATE, Outcome.FAILED

        if node is Node.VALIDATE:
            with Tracer(trace, "validator", attempt=state.draft_attempts) as t:
                result = self.validator.run(state)
                t.record_usage(result.input_tokens, result.output_tokens, result.cost_usd)

                if result.ok:
                    state.final = state.draft
                    state.validation_notes = []
                    t.set(StepStatus.OK, "approved", layer=result.data.get("layer"))
                    return Node.DONE, Outcome.APPROVED

                state.validation_notes = result.data.get("notes", [])
                t.set(
                    StepStatus.REJECTED,
                    f"{len(state.validation_notes)} unsupported claims",
                    layer=result.data.get("layer"),
                    notes=state.validation_notes,
                )

            # Rejection: decide whether another attempt is worth making.
            if state.draft_attempts >= self.max_draft_attempts:
                return Node.DONE, Outcome.UNVALIDATED
            if self.budget.headroom_fraction() < self.min_headroom_for_retry:
                with Tracer(trace, "orchestrator") as t:
                    t.set(
                        StepStatus.SKIPPED,
                        "retry skipped: insufficient budget headroom",
                        headroom=round(self.budget.headroom_fraction(), 3),
                    )
                return Node.DONE, Outcome.UNVALIDATED
            return Node.DRAFT, Outcome.FAILED

        return Node.FAILED, Outcome.FAILED
