from agentdesk.agents.roles import (
    Document,
    DrafterAgent,
    PlannerAgent,
    RetrieverAgent,
    RunState,
    ValidatorAgent,
)
from agentdesk.core.budget import Budget
from agentdesk.corpus import SAMPLE_CORPUS
from agentdesk.llm.client import ScriptedClient


def _state(task="why did gross margin decline"):
    return RunState(task=task, corpus=list(SAMPLE_CORPUS))


def test_planner_returns_steps():
    result = PlannerAgent(ScriptedClient(), Budget()).run(_state())
    assert result.ok and len(result.data["plan"]) >= 2


def test_retriever_costs_a_step_not_tokens():
    budget = Budget()
    state = _state()
    state.plan = ["find margin figures"]
    result = RetrieverAgent(ScriptedClient(), budget).run(state)
    assert result.ok
    assert budget.tokens_used == 0 and budget.steps_used == 1


def test_retriever_reports_failure_on_no_match():
    state = RunState(task="zzz qqq", corpus=[Document("d", "unrelated text")])
    assert not RetrieverAgent(ScriptedClient(), Budget()).run(state).ok


def test_validator_structural_layer_costs_no_tokens():
    budget = Budget()
    state = _state()
    state.retrieved = list(SAMPLE_CORPUS[:2])
    state.draft = "Revenue grew 40% last year."  # number, no citation
    result = ValidatorAgent(ScriptedClient(), budget).run(state)
    assert not result.ok
    assert result.data["layer"] == "structural"
    assert budget.tokens_used == 0


def test_validator_approves_cited_draft():
    state = _state()
    state.retrieved = list(SAMPLE_CORPUS[:2])
    state.draft = "Revenue grew 7.3% [1]. Margin fell [2]."
    assert ValidatorAgent(ScriptedClient(), Budget()).run(state).ok


def test_drafter_includes_rejection_feedback_in_prompt():
    client = ScriptedClient()
    state = _state()
    state.retrieved = list(SAMPLE_CORPUS[:2])
    state.validation_notes = ["The CEO said growth would continue."]
    DrafterAgent(client, Budget()).run(state)
    _system, user = client.calls[-1]
    assert "CEO said growth" in user
