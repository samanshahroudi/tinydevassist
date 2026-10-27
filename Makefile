PYTHON ?= python3

.PHONY: setup setup-train setup-app doctor test lint format format-check check

setup:
	$(PYTHON) -m pip install -e '.[dev]'

setup-train:
	$(PYTHON) -m pip install -e '.[train]'

setup-app:
	$(PYTHON) -m pip install -e '.[app]'

doctor:
	$(PYTHON) -m tinydevassist doctor

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

check: lint format-check test
