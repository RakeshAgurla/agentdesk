"""Budget enforcement.

Agent systems fail expensively. A retry loop that looks reasonable in testing
can burn through a hundred dollars overnight when a validator keeps rejecting a
draft that the drafter keeps reproducing.

The important design decision here: budget is checked *before* each call, not
measured after. Measuring after tells you what you already spent. Checking
before is the only version that actually stops anything.

Three independent ceilings, because they fail differently:
  - tokens: the thing you are actually billed on
  - wall clock: catches a model that is slow rather than expensive
  - step count: catches a cycle in the graph that neither of the above would
    stop quickly enough
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised before a call that would breach a ceiling.

    Deliberately not a return value. A budget breach is not a normal outcome
    the orchestrator should be free to ignore, and making it an exception means
    a caller who forgets to check cannot silently overspend.
    """

    def __init__(self, limit_name: str, used: float, limit: float):
        self.limit_name = limit_name
        self.used = used
        self.limit = limit
        super().__init__(
            f"budget exceeded: {limit_name} would reach {used:.2f} of {limit:.2f}"
        )


@dataclass
class Budget:
    max_tokens: int = 50_000
    max_seconds: float = 120.0
    max_steps: int = 25
    # Prices per million tokens. Defaults are placeholders -- override per model
    # rather than trusting a constant baked into a repo that will go stale.
    input_price_per_mtok: float = 3.0
    output_price_per_mtok: float = 15.0

    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    steps_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.input_price_per_mtok
            + self.output_tokens / 1_000_000 * self.output_price_per_mtok
        )

    def check(self, projected_tokens: int = 0) -> None:
        """Raise if the next call would breach any ceiling.

        projected_tokens is an estimate of the *upcoming* call. It is always
        wrong, which is fine -- it only needs to be wrong in the conservative
        direction, and over-estimating input length is easy.
        """
        if self.steps_used + 1 > self.max_steps:
            raise BudgetExceeded("steps", self.steps_used + 1, self.max_steps)

        elapsed = self.elapsed
        if elapsed > self.max_seconds:
            raise BudgetExceeded("seconds", elapsed, self.max_seconds)

        projected = self.tokens_used + projected_tokens
        if projected > self.max_tokens:
            raise BudgetExceeded("tokens", projected, self.max_tokens)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.tokens_used += input_tokens + output_tokens
        self.steps_used += 1

    def snapshot(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "steps_used": self.steps_used,
            "elapsed_s": round(self.elapsed, 3),
            "cost_usd": round(self.cost_usd, 6),
            "tokens_remaining": max(0, self.max_tokens - self.tokens_used),
            "steps_remaining": max(0, self.max_steps - self.steps_used),
        }

    def headroom_fraction(self) -> float:
        """How much of the tightest budget remains, 0.0 to 1.0.

        The orchestrator uses this to decide whether another retry is worth
        attempting, rather than starting a loop it cannot finish.
        """
        fractions = [
            1.0 - self.tokens_used / self.max_tokens if self.max_tokens else 1.0,
            1.0 - self.elapsed / self.max_seconds if self.max_seconds else 1.0,
            1.0 - self.steps_used / self.max_steps if self.max_steps else 1.0,
        ]
        return max(0.0, min(fractions))
