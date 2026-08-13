.PHONY: install test lint run demo clean

PY ?= python3
export PYTHONPATH := src

install:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q

lint:
	ruff check src tests

run:
	$(PY) -m agentdesk.cli "why did gross margin decline"

demo:
	@echo "--- happy path ---"
	@$(PY) -m agentdesk.cli "why did gross margin decline" || true
	@echo "\n--- rejection then recovery ---"
	@$(PY) -m agentdesk.cli "why did gross margin decline" --failure-mode unsupported_claim || true
	@echo "\n--- fails to converge ---"
	@$(PY) -m agentdesk.cli "why did gross margin decline" --failure-mode always_bad || true
	@echo "\n--- budget exhausted ---"
	@$(PY) -m agentdesk.cli "why did gross margin decline" --max-tokens 400 || true

clean:
	rm -rf .pytest_cache **/__pycache__ trace.json
