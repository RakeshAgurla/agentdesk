from agentdesk.core.budget import Budget
from agentdesk.core.orchestrator import Orchestrator, Outcome
from agentdesk.corpus import SAMPLE_CORPUS
from agentdesk.llm.client import ScriptedClient

TASK = "why did gross margin decline"


def _run(client=None, budget=None, **kw):
    return Orchestrator(
        client=client or ScriptedClient(), budget=budget or Budget(), **kw
    ).run(TASK, list(SAMPLE_CORPUS))


def test_happy_path_approves():
    result = _run()
    assert result.outcome is Outcome.APPROVED and result.trustworthy


def test_rejection_triggers_retry_then_recovers():
    result = _run(ScriptedClient(failure_mode="unsupported_claim"))
    assert result.outcome is Outcome.APPROVED
    assert result.trace.attempts_for("drafter") == 2
    assert len(result.trace.rejections()) == 1


def test_retry_ceiling_stops_the_loop():
    result = _run(ScriptedClient(failure_mode="always_bad"), max_draft_attempts=3)
    assert result.outcome is Outcome.UNVALIDATED
    assert not result.trustworthy
    assert result.trace.attempts_for("drafter") == 3


def test_unvalidated_still_returns_the_draft():
    result = _run(ScriptedClient(failure_mode="always_bad"))
    assert result.answer and result.validation_notes


def test_budget_stops_run_cleanly():
    result = _run(budget=Budget(max_tokens=400))
    assert result.outcome is Outcome.BUDGET_EXHAUSTED
    assert not result.trustworthy


def test_low_headroom_skips_retry():
    # Enough budget to draft once, not enough to retry.
    result = _run(
        ScriptedClient(failure_mode="always_bad"),
        budget=Budget(max_tokens=1400),
        max_draft_attempts=5,
    )
    assert result.outcome in (Outcome.UNVALIDATED, Outcome.BUDGET_EXHAUSTED)
    assert result.trace.attempts_for("drafter") < 5


def test_trace_covers_every_step():
    result = _run()
    agents = [s.agent for s in result.trace.steps]
    assert agents == ["planner", "retriever", "drafter", "validator"]


def test_budget_snapshot_reported():
    result = _run()
    assert result.budget["tokens_used"] > 0
    assert result.budget["cost_usd"] >= 0
