"""CLI entry point.

    python -m agentdesk.cli "why did gross margin decline"
    python -m agentdesk.cli "..." --failure-mode unsupported_claim
    python -m agentdesk.cli "..." --max-tokens 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentdesk.core.budget import Budget
from agentdesk.core.orchestrator import Orchestrator, Outcome
from agentdesk.corpus import SAMPLE_CORPUS
from agentdesk.llm.client import build_client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--backend", default=None, help="scripted | anthropic | openai")
    parser.add_argument("--failure-mode", default=None,
                        help="unsupported_claim | always_bad (scripted backend only)")
    parser.add_argument("--max-tokens", type=int, default=50_000)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--trace-out", type=Path, default=None)
    args = parser.parse_args()

    kwargs = {}
    backend = args.backend or "scripted"
    if backend == "scripted" and args.failure_mode:
        kwargs["failure_mode"] = args.failure_mode

    orchestrator = Orchestrator(
        client=build_client(backend, **kwargs),
        budget=Budget(max_tokens=args.max_tokens, max_steps=args.max_steps),
        max_draft_attempts=args.max_attempts,
    )

    result = orchestrator.run(args.task, SAMPLE_CORPUS)

    print(result.trace.render())
    print()
    print(f"outcome: {result.outcome.value}  trustworthy: {result.trustworthy}")
    if result.validation_notes:
        print("outstanding validation notes:")
        for note in result.validation_notes:
            print(f"  - {note}")
    print()
    print(result.answer or "(no answer produced)")

    if args.trace_out:
        result.trace.save(args.trace_out)
        print(f"\ntrace written to {args.trace_out}")

    return 0 if result.outcome is Outcome.APPROVED else 1


if __name__ == "__main__":
    sys.exit(main())
