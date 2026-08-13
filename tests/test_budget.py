import time

import pytest

from agentdesk.core.budget import Budget, BudgetExceeded


def test_check_blocks_before_spending():
    b = Budget(max_tokens=100)
    with pytest.raises(BudgetExceeded) as exc:
        b.check(projected_tokens=150)
    assert exc.value.limit_name == "tokens"
    assert b.tokens_used == 0  # nothing spent


def test_check_allows_within_limit():
    Budget(max_tokens=100).check(projected_tokens=50)


def test_step_ceiling():
    b = Budget(max_steps=2)
    b.record(1, 1)
    b.record(1, 1)
    with pytest.raises(BudgetExceeded) as exc:
        b.check()
    assert exc.value.limit_name == "steps"


def test_wall_clock_ceiling():
    b = Budget(max_seconds=0.01)
    time.sleep(0.02)
    with pytest.raises(BudgetExceeded) as exc:
        b.check()
    assert exc.value.limit_name == "seconds"


def test_cost_uses_separate_input_output_prices():
    b = Budget(input_price_per_mtok=3.0, output_price_per_mtok=15.0)
    b.record(1_000_000, 1_000_000)
    assert b.cost_usd == pytest.approx(18.0)


def test_headroom_reflects_tightest_constraint():
    b = Budget(max_tokens=1000, max_steps=10)
    b.record(900, 0)
    assert b.headroom_fraction() == pytest.approx(0.1)


def test_headroom_never_negative():
    b = Budget(max_tokens=10)
    b.record(100, 0)
    assert b.headroom_fraction() == 0.0
