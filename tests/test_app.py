"""Bounded demo and deployment packaging tests without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import app
import pytest
from deploy_hf import deployment_manifest


def test_demo_response_controls(monkeypatch):
    calls = []

    class Stub:
        def generate(self, prompt, *, max_new_tokens, temperature):
            calls.append((prompt, max_new_tokens, temperature))
            return SimpleNamespace(response="check the traceback")

    monkeypatch.setattr(app, "load_generator", lambda: Stub())
    assert app.respond("  Why does import fail?  ", 0, 96) == "check the traceback"
    assert calls == [("Why does import fail?", 96, 0.0)]
    with pytest.raises(ValueError):
        app.respond(" ", 0, 96)
    with pytest.raises(ValueError):
        app.respond("x", 0, 999)


def test_deployment_manifest_contains_only_curated_artifacts(tmp_path):
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ):
        (tmp_path / filename).touch()
    model_files, space_files = deployment_manifest(tmp_path)
    assert "training_args.bin" not in model_files
    assert "adapter_model.safetensors" in model_files
    assert "app.py" in space_files
