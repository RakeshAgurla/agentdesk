# agentdesk

A multi-agent pipeline that can fail properly.

Most public agent repos demonstrate the happy path: three agents pass strings
around, the output looks plausible, and there is no answer to "what happens when
the drafter invents a number" or "what stops this from costing $400 overnight".
This one is built around those questions.

```bash
git clone <this-repo> && cd agentdesk
pip install -e ".[dev]"
python -m agentdesk.cli "why did gross margin decline"
```

No API key required. The default backend is a scripted mock, so the full
pipeline — including its failure paths — runs offline in under a second.

---

## The four paths

**Happy path.** Plan, retrieve, draft, validate, approve.

```
run edb1aa6c197c  outcome=approved
4 steps | 499 tokens | $0.0023

   1. [ok  ] planner      attempt 1   70tok  3 steps
   2. [ok  ] retriever    attempt 1    0tok  2 docs
   3. [ok  ] drafter      attempt 1  219tok  150 chars
   4. [ok  ] validator    attempt 1  210tok  approved
```

**Rejection and recovery.** The validator rejects an uncited claim, the specific
offending sentence is fed back to the drafter, and the second attempt passes.

```bash
python -m agentdesk.cli "why did gross margin decline" --failure-mode unsupported_claim
```
```
   3. [ok  ] drafter      attempt 1  212tok  120 chars
   4. [rej ] validator    attempt 1    0tok  1 unsupported claims
   5. [ok  ] drafter      attempt 2  254tok  150 chars
   6. [ok  ] validator    attempt 2  210tok  approved
```

Note the rejection cost **zero tokens** — see "two-layer validation" below.

**Failure to converge.** The drafter cannot fix the problem. After three
attempts the run stops and returns `unvalidated` with the outstanding notes
attached. The draft is still returned, but `trustworthy` is `False`.

```bash
python -m agentdesk.cli "..." --failure-mode always_bad
```

**Budget exhaustion.** The ceiling is hit before a call, not after.

```bash
python -m agentdesk.cli "..." --max-tokens 400
```
```
   1. [err ] planner  BudgetExceeded: tokens would reach 441.00 of 400.00
   2. [budg] orchestrator  tokens limit reached
```

Nothing was spent. That is the point.

---

## Three design decisions

### Budget is checked before each call

Measuring spend after the fact tells you what you already lost. `Budget.check()`
projects the cost of the *next* call and raises before it happens.

Three independent ceilings, because they fail differently: **tokens** (what you
are billed on), **wall clock** (catches a slow model rather than an expensive
one), and **step count** (catches a cycle in the graph faster than either).

There is also a headroom check before each retry. Starting an attempt there is
not enough budget to finish costs the same as one that completes and produces
nothing, so the orchestrator declines it and records why.

### Two-layer validation

A deterministic citation check runs first: any sentence containing a figure or
an assertion verb but no `[n]` marker is flagged, with no model call at all.
This catches the most common failure — uncited numbers — reliably and for free.

Only if that passes does the model layer run, which catches claims that are
cited but not actually supported by the cited text.

Ordering matters. Running the expensive check first would pay for a model call
to find something a regex finds in microseconds.

### The orchestrator is a plain state machine

Named nodes and an explicit transition function. No framework.

LangGraph is a reasonable alternative and at larger scale it earns its keep —
persistence, streaming, human-in-the-loop checkpoints. At this size it would
hide the only part worth showing. The control flow *is* the engineering here:
what happens when validation fails twice, what happens when the budget runs out
mid-retry, what the system returns when it cannot finish.

The trade-off is real: this does not persist state across process restarts.
Needing that is the point at which to adopt a framework rather than reimplement
one.

---

## Outcomes are distinct

```python
class Outcome(str, Enum):
    APPROVED          # validated, safe to show
    UNVALIDATED       # draft exists, never passed validation
    BUDGET_EXHAUSTED  # stopped early
    FAILED            # no draft at all
```

Collapsing these into one string is the easy mistake. A caller needs to
distinguish "here is a verified answer" from "here is an answer nothing
verified" — `RunResult.trustworthy` is only `True` for the first.

---

## Architecture

```
task ──► planner ──► retriever ──► drafter ──► validator ──► approved
                                      ▲            │
                                      └── reject ──┤ (max 3, budget permitting)
                                                   │
                                                   ▼
                                              unvalidated
```

Agents never decide whether the pipeline continues — they report what happened
and the orchestrator decides. Agents that control their own retries produce
nested loops nobody can reason about, and the budget becomes unenforceable
because no single place knows the total.

State is a single mutable `RunState` passed between nodes rather than agents
calling each other. It makes the data flow inspectable, each node independently
testable, and adding a node does not require touching its neighbours.

```
src/agentdesk/
├── core/
│   ├── budget.py        pre-call enforcement, three ceilings
│   ├── trace.py         structured per-step record
│   └── orchestrator.py  the state machine
├── agents/roles.py      planner, retriever, drafter, validator
├── llm/client.py        scripted | anthropic | openai
├── corpus.py            deterministic fixture
└── cli.py
```

## Traces

Every step records tokens, latency, cost, attempt number, and outcome. Traces
serialise to JSON so a failing run can be attached to a bug report:

```bash
python -m agentdesk.cli "..." --trace-out trace.json
```

This is not a logging wrapper. The orchestrator reads the trace to decide
whether a retry is worth making.

## Testing

```bash
pytest tests -q     # 26 tests, ~0.1s
```

The scripted client can be told to misbehave on cue (`failure_mode`), so CI
exercises rejection, retry, retry-ceiling, and budget-exhaustion on every run.
A mock that always returns good output would test only the path that was never
going to fail.

## Known limitations

- The retriever is lexical overlap, not semantic. Adequate for demonstrating the
  agent layer; a real deployment would use a proper retrieval backend.
- No state persistence across restarts.
- Agents run sequentially. Planner and retriever could overlap, but the added
  concurrency would make budget accounting materially harder to get right.
- Token estimation for pre-call projection is a character heuristic. Wrong in
  the conservative direction, which is what the check needs, but not exact.
- The model validation layer is only as good as the model. The deterministic
  layer is the one with guarantees.

## License

MIT
