"""Tests for dependency-free project diagnostics."""

import json

from tinydevassist import DEFAULT_BASE_MODEL, DEFAULT_SEED, __version__
from tinydevassist.__main__ import main
from tinydevassist.environment import OPTIONAL_MODULES, collect_environment, environment_json


def test_default_configuration_is_explicit() -> None:
    assert DEFAULT_SEED == 42
    assert DEFAULT_BASE_MODEL == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert __version__ == "0.1.0"


def test_doctor_is_machine_readable_and_offline() -> None:
    result = collect_environment()
    assert result["base_model_candidate"] == DEFAULT_BASE_MODEL
    assert set(result["optional_dependencies_present"]) == set(OPTIONAL_MODULES)
    assert all(isinstance(v, bool) for v in result["optional_dependencies_present"].values())
    assert json.loads(environment_json()) == result


def test_doctor_cli(capsys) -> None:
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["project_version"] == __version__
