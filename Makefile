.PHONY: install demo dashboard lint format-check typecheck test check

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

check: lint format-check typecheck test
