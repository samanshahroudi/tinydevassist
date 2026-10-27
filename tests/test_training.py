"""Offline LoRA preparation tests; these do not execute a training experiment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tinydevassist.data import prepare_dataset
from tinydevassist.training import (
    load_training_config,
    load_verified_splits,
    pad_supervised_batch,
    tokenize_supervised_example,
    training_plan,
    verify_baseline,
)
from train_lora import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    config = load_training_config(ROOT / "configs/train_lora.json")
    source = ROOT / "data/samples/developer_support_v1.jsonl"
    processed = tmp_path / "processed"
    prepare_dataset(source, processed, seed=42)
    manifest, splits = load_verified_splits(processed, source, config)
    return config, source, processed, manifest, splits


def test_dry_run_reports_real_hashes_and_counts(inputs):
    config, _, _, manifest, splits = inputs
    plan = training_plan(config, manifest)
    assert plan["status"] == "dry_run_only"
    assert plan["training_executed"] is False
    assert plan["split_counts"] == {"train": 24, "validation": 6, "test": 6}
    assert plan["estimated_optimizer_steps"] == 6
    assert plan["requires_pinned_revision"] is True
    assert len(splits["train"]) == 24
    assert plan["dataset_source_sha256"] == manifest["source_sha256"]


def test_reject_tampered_split(inputs):
    config, source, processed, _, _ = inputs
    with (processed / "train.jsonl").open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_splits(processed, source, config)


def test_reject_source_drift(inputs, tmp_path):
    config, source, processed, _, _ = inputs
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source dataset hash"):
        load_verified_splits(processed, changed, config)


def test_reject_config_seed_drift(inputs):
    config, source, processed, _, _ = inputs
    with pytest.raises(ValueError, match="seed"):
        load_verified_splits(processed, source, replace(config, seed=100))


@pytest.mark.parametrize(
    "field,value",
    [("seed", True), ("max_sequence_length", 0), ("revision", ""), ("base_model_id", "")],
)
def test_reject_invalid_config_fields(tmp_path, field, value):
    payload = json.loads((ROOT / "configs/train_lora.json").read_text())
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_training_config(path)


def test_reject_duplicate_lora_target_modules(tmp_path):
    payload = json.loads((ROOT / "configs/train_lora.json").read_text())
    payload["lora"]["target_modules"] = ["q_proj", "q_proj"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="target_modules"):
        load_training_config(path)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        if add_generation_prompt:
            assert len(messages) == 2
            return [1, 2, 3]
        assert len(messages) == 3
        return [1, 2, 3, 4, 5]


def test_assistant_only_loss_and_right_padding():
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    first = tokenize_supervised_example(FakeTokenizer(), messages, 512)
    assert first == {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, -100, 4, 5],
    }
    second = {"input_ids": [8, 9], "attention_mask": [1, 1], "labels": [-100, 9]}
    padded = pad_supervised_batch([first, second], pad_token_id=0)
    assert padded["input_ids"][1] == [8, 9, 0, 0, 0]
    assert padded["attention_mask"][1] == [1, 1, 0, 0, 0]
    assert padded["labels"][1] == [-100, 9, -100, -100, -100]


def test_reject_overlength_sample():
    messages = [{"role": role, "content": role} for role in ("system", "user", "assistant")]
    with pytest.raises(ValueError, match="exceeds"):
        tokenize_supervised_example(FakeTokenizer(), messages, 4)


def test_reject_mismatched_chat_template_prefix():
    class BadTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [1, 2, 3] if add_generation_prompt else [9, 2, 3, 4]

    messages = [{"role": role, "content": role} for role in ("system", "user", "assistant")]
    with pytest.raises(ValueError, match="prefix"):
        tokenize_supervised_example(BadTokenizer(), messages, 512)


def test_execute_rejects_floating_revision_before_model_load(inputs, tmp_path, capsys):
    _, source, processed, _, _ = inputs
    result = main([
        "--source", str(source), "--processed-dir", str(processed),
        "--output-dir", str(tmp_path / "adapter"), "--report", str(tmp_path / "report.json"),
        "--execute",
    ])
    assert result == 1
    assert "pinned" in capsys.readouterr().err
    assert not (tmp_path / "adapter").exists()


def test_execute_rejects_missing_baseline_without_importing_transformers(inputs, tmp_path, capsys):
    _, source, processed, _, _ = inputs
    result = main([
        "--source", str(source), "--processed-dir", str(processed),
        "--baseline-report", str(tmp_path / "missing.json"),
        "--revision", "a" * 40, "--execute",
    ])
    assert result == 1
    assert "missing.json" in capsys.readouterr().err


def test_verify_baseline_requires_matching_revision_and_full_test_coverage(inputs, tmp_path):
    config, _, _, manifest, _ = inputs
    config = replace(config, revision="a" * 40)
    path = tmp_path / "baseline.json"
    baseline = {
        "status": "measured",
        "model_id": config.base_model_id,
        "resolved_revision": config.revision,
        "requested_revision": config.revision,
        "input_sha256": manifest["splits"]["test"]["sha256"],
        "results": [
            {
                "id": row_id,
                "response": "test response",
                "metrics": {"latency_seconds": 0.1, "generated_tokens": 5},
            }
            for row_id in manifest["splits"]["test"]["ids"]
        ],
    }
    path.write_text(json.dumps(baseline))
    verify_baseline(path, config, manifest)
    baseline["results"].pop()
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="every held-out"):
        verify_baseline(path, config, manifest)
    baseline["results"] = [
        {
            "id": row_id,
            "response": "test response",
            "metrics": {"latency_seconds": 0.1, "generated_tokens": 5},
        }
        for row_id in manifest["splits"]["test"]["ids"]
    ]
    baseline["resolved_revision"] = "b" * 40
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="matching test hash"):
        verify_baseline(path, config, manifest)


def test_training_never_overwrites_existing_adapter(inputs, tmp_path):
    from tinydevassist.training import run_training

    config, _, _, manifest, splits = inputs
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    (output_dir / "adapter_config.json").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        run_training(config, manifest, splits, output_dir)
    assert (output_dir / "adapter_config.json").read_text() == "keep me"
