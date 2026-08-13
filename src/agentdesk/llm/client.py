"""LLM client.

One interface, three backends. The scripted mock is what CI runs: it returns
canned responses keyed by which agent is asking, including deliberately bad
ones so the validation path is exercised on every test run.

That last part matters. A mock that always returns good output tests only the
happy path, which is the path that was never going to fail.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

    def cost(self, in_price: float, out_price: float) -> float:
        return (
            self.input_tokens / 1_000_000 * in_price
            + self.output_tokens / 1_000_000 * out_price
        )


def estimate_tokens(text: str) -> int:
    """~4 chars per token. Used for pre-call budget projection only.

    Wrong in the safe direction on English prose, which is what pre-call
    checking needs -- an underestimate would let a call through that should
    have been blocked.
    """
    return max(1, len(text) // 4)


class LLMClient(ABC):
    model: str

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        ...


class ScriptedClient(LLMClient):
    """Deterministic mock.

    Responses are chosen by matching against the system prompt, so each agent
    gets output shaped like what it would really receive. `failure_mode` forces
    specific bad outputs to test rejection, retry, and budget paths without
    needing a real model to misbehave on cue.
    """

    model = "scripted-mock"

    def __init__(self, failure_mode: str | None = None, max_bad_attempts: int = 1):
        self.failure_mode = failure_mode
        self.max_bad_attempts = max_bad_attempts
        self.calls: list[tuple[str, str]] = []
        self._bad_so_far = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        self.calls.append((system, user))
        text = self._respond(system, user)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(system + user),
            output_tokens=estimate_tokens(text),
            model=self.model,
        )

    def _respond(self, system: str, user: str) -> str:
        role = self._role(system)

        if role == "planner":
            return (
                "1. Retrieve revenue and margin figures\n"
                "2. Retrieve stated drivers of the margin change\n"
                "3. Draft a summary citing both"
            )

        if role == "drafter":
            # Emit a bad draft for the first N attempts so the validator has
            # something real to reject, then recover.
            if self.failure_mode == "unsupported_claim" and self._bad_so_far < self.max_bad_attempts:
                self._bad_so_far += 1
                return (
                    "Revenue grew 7.3% to $4.82 billion [1]. Margin fell to 26.4% [1]. "
                    "The company also plans a $2 billion buyback next year."
                )
            if self.failure_mode == "always_bad":
                return "Revenue grew substantially. The CEO said growth would continue."
            return (
                "Revenue grew 7.3% to $4.82 billion [1]. "
                "Gross margin declined to 26.4% from 28.1% [1], attributed to higher "
                "raw material costs and tariff expense [2]."
            )

        if role == "validator":
            # The user message contains excerpts AND the draft. Only the draft
            # is under review -- scanning the excerpts flags the source material
            # itself as uncited, which is how this was wrong the first time.
            draft = user.split("Draft:", 1)[-1] if "Draft:" in user else user
            unsupported = self._find_unsupported(draft)
            if unsupported:
                return "REJECT\n" + "\n".join(f"- {c}" for c in unsupported)
            return "APPROVE"

        return "OK"

    @staticmethod
    def _role(system: str) -> str:
        lowered = system.lower()
        for role in ("planner", "drafter", "validator"):
            if role in lowered:
                return role
        return "unknown"

    @staticmethod
    def _find_unsupported(draft: str) -> list[str]:
        """Flag sentences carrying a factual claim but no citation marker."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
        flagged = []
        for sentence in sentences:
            has_citation = bool(re.search(r"\[\d+\]", sentence))
            has_claim = bool(re.search(r"\d|said|plans|expects|will", sentence, re.IGNORECASE))
            if has_claim and not has_citation:
                flagged.append(sentence)
        return flagged


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )


def build_client(backend: str | None = None, **kwargs) -> LLMClient:
    backend = backend or os.getenv("AGENTDESK_LLM_BACKEND", "scripted")
    if backend == "scripted":
        return ScriptedClient(**kwargs)
    if backend == "anthropic":
        return AnthropicClient(**kwargs)
    if backend == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
