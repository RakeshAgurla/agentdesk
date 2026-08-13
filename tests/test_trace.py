import pytest

from agentdesk.core.trace import RunTrace, StepStatus, Tracer


def test_tracer_records_success():
    run = RunTrace(task="t")
    with Tracer(run, "planner") as t:
        t.record_usage(10, 20, 0.5)
        t.set(StepStatus.OK, "done")
    assert len(run.steps) == 1
    assert run.steps[0].status is StepStatus.OK
    assert run.total_tokens == 30


def test_tracer_records_exception_and_reraises():
    run = RunTrace(task="t")
    with pytest.raises(ValueError), Tracer(run, "drafter"):
        raise ValueError("boom")
    assert run.steps[0].status is StepStatus.ERROR
    assert "boom" in run.steps[0].error


def test_attempts_and_rejections_counted():
    run = RunTrace(task="t")
    for i in (1, 2):
        with Tracer(run, "validator", attempt=i) as t:
            t.set(StepStatus.REJECTED, "no")
    assert run.attempts_for("validator") == 2
    assert len(run.rejections()) == 2


def test_trace_serialises():
    run = RunTrace(task="t")
    with Tracer(run, "planner") as t:
        t.set(StepStatus.OK)
    d = run.to_dict()
    assert d["step_count"] == 1 and d["steps"][0]["status"] == "ok"


def test_render_is_readable():
    run = RunTrace(task="t")
    with Tracer(run, "planner") as t:
        t.set(StepStatus.OK, "3 steps")
    assert "planner" in run.render()
