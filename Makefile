PYTHON ?= python3

.PHONY: setup setup-train setup-app setup-notebooks baseline train-plan doctor prepare test lint format format-check check

setup:
	$(PYTHON) -m pip install -e '.[dev]'

setup-train:
	$(PYTHON) -m pip install -e '.[train]'

setup-app:
	$(PYTHON) -m pip install -e '.[app]'

setup-notebooks:
	$(PYTHON) -m pip install -e '.[notebooks]'

baseline:
	$(PYTHON) inference.py --device cpu

train-plan:
	$(PYTHON) train_lora.py

doctor:
	$(PYTHON) -m tinydevassist doctor

prepare:
	$(PYTHON) prepare_data.py --seed 42

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

check: lint format-check test
