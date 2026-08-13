"""The agents.

Four roles, each with one job and an explicit contract about what it returns.

The design principle throughout: an agent never decides whether the pipeline
continues. It reports what happened and the orchestrator decides. Agents that
control their own retries end up with nested loops nobody can reason about, and
the budget becomes unenforceable because no single place knows the total.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentdesk.core.budget import Budget
from agentdesk.llm.client import LLMClient, estimate_tokens


@dataclass
class Document:
    doc_id: str
    text: str
    source: str = ""

    def cite(self, n: int) -> str:
        return f"[{n}] ({self.source}) {self.text}"


@dataclass
class AgentResult:
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Agent(ABC):
    name: str
    system_prompt: str

    def __init__(self, client: LLMClient, budget: Budget):
        self.client = client
        self.budget = budget

    def _call(self, user: str, max_tokens: int = 1024) -> tuple[str, int, int, float]:
        """Budget-checked LLM call.

        The check happens here rather than in the orchestrator so that no agent
        can bypass it, including ones added later by someone who did not read
        this docstring.
        """
        projected = estimate_tokens(self.system_prompt + user) + max_tokens
        self.budget.check(projected_tokens=projected)

        response = self.client.complete(self.system_prompt, user, max_tokens)
        cost = response.cost(
            self.budget.input_price_per_mtok, self.budget.output_price_per_mtok
        )
        self.budget.record(response.input_tokens, response.output_tokens)
        return response.text, response.input_tokens, response.output_tokens, cost

    @abstractmethod
    def run(self, state: RunState) -> AgentResult:
        ...


@dataclass
class RunState:
    """Everything the pipeline knows, passed between nodes.

    A single mutable state object rather than agents calling each other
    directly. It makes the data flow inspectable, makes each node independently
    testable, and means adding a node does not require touching the ones around
    it.
    """

    task: str
    corpus: list[Document] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    retrieved: list[Document] = field(default_factory=list)
    draft: str = ""
    validation_notes: list[str] = field(default_factory=list)
    final: str = ""
    draft_attempts: int = 0

    def context_block(self) -> str:
        return "\n\n".join(d.cite(i) for i, d in enumerate(self.retrieved, 1))


class PlannerAgent(Agent):
    name = "planner"
    system_prompt = (
        "You are the planner. Break the task into 2-4 concrete retrieval and "
        "drafting steps. Output one step per line, numbered. No preamble."
    )

    def run(self, state: RunState) -> AgentResult:
        text, tin, tout, cost = self._call(f"Task: {state.task}", max_tokens=400)
        steps = [
            re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]
        if not steps:
            return AgentResult(
                ok=False, reason="planner produced no steps",
                input_tokens=tin, output_tokens=tout, cost_usd=cost,
            )
        return AgentResult(
            ok=True, output=text, data={"plan": steps},
            input_tokens=tin, output_tokens=tout, cost_usd=cost,
        )


class RetrieverAgent(Agent):
    """Deliberately not an LLM call.

    Retrieval here is lexical overlap against the corpus. Using a model to pick
    documents would cost tokens and latency to do worse than BM25 does for free.
    The interesting agent behaviour is downstream; this node exists to supply
    grounded context and to keep the retrieval boundary explicit.
    """

    name = "retriever"
    system_prompt = "retriever"

    def __init__(self, client: LLMClient, budget: Budget, top_k: int = 3):
        super().__init__(client, budget)
        self.top_k = top_k

    def run(self, state: RunState) -> AgentResult:
        query_terms = set(_tokens(state.task + " " + " ".join(state.plan)))
        scored: list[tuple[float, Document]] = []
        for doc in state.corpus:
            doc_terms = set(_tokens(doc.text))
            if not doc_terms:
                continue
            overlap = len(query_terms & doc_terms) / len(query_terms | doc_terms)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda pair: -pair[0])
        hits = [doc for _score, doc in scored[: self.top_k]]

        self.budget.steps_used += 1  # costs a step, not tokens

        if not hits:
            return AgentResult(ok=False, reason="no documents matched the task")
        return AgentResult(
            ok=True,
            output=f"{len(hits)} documents",
            data={"retrieved": hits, "top_score": round(scored[0][0], 3)},
        )


class DrafterAgent(Agent):
    name = "drafter"
    system_prompt = (
        "You are the drafter. Write a short factual summary answering the task "
        "using ONLY the numbered excerpts provided. Cite every factual claim "
        "with its excerpt number in square brackets. Do not add information "
        "that is not in the excerpts."
    )

    def run(self, state: RunState) -> AgentResult:
        feedback = ""
        if state.validation_notes:
            feedback = (
                "\n\nA previous draft was rejected for these unsupported claims. "
                "Remove or cite them:\n"
                + "\n".join(f"- {n}" for n in state.validation_notes)
            )

        user = f"Task: {state.task}\n\nExcerpts:\n{state.context_block()}{feedback}"
        text, tin, tout, cost = self._call(user, max_tokens=800)

        if not text.strip():
            return AgentResult(
                ok=False, reason="empty draft",
                input_tokens=tin, output_tokens=tout, cost_usd=cost,
            )
        return AgentResult(
            ok=True, output=text, data={"draft": text},
            input_tokens=tin, output_tokens=tout, cost_usd=cost,
        )


class ValidatorAgent(Agent):
    """The agent that can say no.

    Most published agent pipelines have a "reviewer" that approves everything,
    because rejection is only useful if something downstream acts on it. Here a
    rejection routes back to the drafter with the specific offending sentences
    attached, and the orchestrator counts attempts against a ceiling.

    Validation is two-layered on purpose: a deterministic citation check that
    needs no model, then the model's judgement. The deterministic layer catches
    the common failure (uncited numbers) reliably and for free; the model
    catches claims that are cited but not actually supported by the cited text.
    """

    name = "validator"
    system_prompt = (
        "You are the validator. Check that every factual claim in the draft is "
        "supported by the numbered excerpts. Respond APPROVE if all claims are "
        "supported. Otherwise respond REJECT followed by one line per "
        "unsupported claim."
    )

    def run(self, state: RunState) -> AgentResult:
        structural = self._uncited_claims(state.draft)
        if structural:
            self.budget.steps_used += 1
            return AgentResult(
                ok=False,
                reason="uncited factual claims",
                data={"notes": structural, "layer": "structural"},
            )

        user = f"Excerpts:\n{state.context_block()}\n\nDraft:\n{state.draft}"
        text, tin, tout, cost = self._call(user, max_tokens=500)

        if text.strip().upper().startswith("APPROVE"):
            return AgentResult(
                ok=True, output=text, data={"layer": "model"},
                input_tokens=tin, output_tokens=tout, cost_usd=cost,
            )

        notes = [
            line.lstrip("- ").strip()
            for line in text.splitlines()[1:]
            if line.strip()
        ]
        return AgentResult(
            ok=False, reason="model rejected draft",
            data={"notes": notes or ["unspecified"], "layer": "model"},
            input_tokens=tin, output_tokens=tout, cost_usd=cost,
        )

    @staticmethod
    def _uncited_claims(draft: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
        flagged = []
        for sentence in sentences:
            has_citation = bool(re.search(r"\[\d+\]", sentence))
            has_claim = bool(re.search(r"\d|said|plans|expects|will", sentence, re.IGNORECASE))
            if has_claim and not has_citation:
                flagged.append(sentence)
        return flagged


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "what", "how", "why", "did", "was", "were", "is", "are", "that", "this",
}


def _tokens(text: str) -> Sequence[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
