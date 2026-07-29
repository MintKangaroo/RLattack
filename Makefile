.PHONY: install demo dashboard lint format-check typecheck test audit check

install:
	python3 -m pip install -e ".[dev]"

demo:
	python3 -m rlattack demo

dashboard:
	python3 -m rlattack dashboard

lint:
	python3 -m ruff check .

format-check:
	python3 -m ruff format --check .

typecheck:
	python3 -m mypy .

test:
	python3 -m pytest

audit:
	python3 -m pip_audit . --skip-editable

check: lint format-check typecheck test
