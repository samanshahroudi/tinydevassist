"""Bounded demo and deployment packaging tests without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import app
import pytest
from deploy_hf import deployment_manifest, verify_matching_model_files
from deploy_zerogpu import zerogpu_manifest


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


def test_demo_can_use_an_eagerly_loaded_generator(monkeypatch):
    class Stub:
        def generate(self, prompt, *, max_new_tokens, temperature):
            return SimpleNamespace(response=f"{prompt}:{max_new_tokens}:{temperature}")

    monkeypatch.setattr(app, "load_generator", lambda: pytest.fail("unexpected model load"))
    assert app.respond("  Git?  ", 0, 64, generator=Stub()) == "Git?:64:0.0"


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


def test_space_only_recovery_rejects_changed_model_files(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (local / "adapter_config.json").write_bytes(b"abc")
    (remote / "adapter_config.json").write_bytes(b"abd")
    with pytest.raises(ValueError, match="differs from local artifact"):
        verify_matching_model_files(
            {"adapter_config.json": local / "adapter_config.json"},
            lambda name: remote / name,
        )


def test_zerogpu_manifest_excludes_docker_package():
    files = zerogpu_manifest()
    assert files["app.py"].name == "app.py"
    assert files["core_app.py"].name == "app.py"
    assert "Dockerfile" not in files
