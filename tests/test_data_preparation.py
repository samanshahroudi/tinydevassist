"""Offline deterministic preparation and prompt-formatting regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tinydevassist.data import (
    _deduplicate,
    _parse_rows,
    normalize_text,
    prepare_dataset,
    prompt_messages,
    question_key,
    training_messages,
)

SOURCE = Path(__file__).resolve().parents[1] / "data/samples/developer_support_v1.jsonl"


def test_normalization_and_role_formatting() -> None:
    assert normalize_text("  Check\n  git\tstatus  ") == "Check git status"
    assert question_key("  GIT  STATUS ") == question_key("git status")
    prompt = prompt_messages(" How do I check Git status? ")
    assert [message["role"] for message in prompt] == ["system", "user"]
    assert prompt[1]["content"] == "How do I check Git status?"
    messages = training_messages("question", " answer ")
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == "answer"
    with pytest.raises(ValueError, match="blank"):
        prompt_messages("   ")
    with pytest.raises(ValueError, match="blank"):
        training_messages("question", "  ")


def test_deterministic_stratified_splits_and_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = prepare_dataset(SOURCE, first, seed=42)
    repeat = prepare_dataset(SOURCE, second, seed=42)
    assert manifest == repeat
    assert manifest["input_count"] == manifest["retained_count"] == 36
    assert {name: data["count"] for name, data in manifest["splits"].items()} == {
        "train": 24,
        "validation": 6,
        "test": 6,
    }
    expected_per_category = {"train": 4, "validation": 1, "test": 1}
    for name, part in manifest["splits"].items():
        assert len(part["categories"]) == 6
        assert set(part["categories"].values()) == {expected_per_category[name]}
    assert sorted(path.name for path in first.iterdir()) == [
        "manifest.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    ]
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    all_ids: list[str] = []
    all_questions: list[str] = []
    for name in ("train", "validation", "test"):
        for line in (first / f"{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            all_ids.append(row["id"])
            all_questions.append(question_key(row["instruction"]))
            assert row["messages"] == training_messages(row["instruction"], row["response"])
    assert len(all_ids) == len(set(all_ids)) == 36
    assert len(all_questions) == len(set(all_questions)) == 36


def test_source_order_does_not_change_split_bytes(tmp_path: Path) -> None:
    original = prepare_dataset(SOURCE, tmp_path / "original")
    rows = SOURCE.read_text(encoding="utf-8").splitlines()
    reordered = tmp_path / "reversed.jsonl"
    reordered.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    changed = prepare_dataset(reordered, tmp_path / "reordered")
    assert original["source_sha256"] != changed["source_sha256"]
    for split in ("train", "validation", "test"):
        assert original["splits"][split] == changed["splits"][split]
        assert (tmp_path / "original" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "reordered" / f"{split}.jsonl"
        ).read_bytes()


def test_duplicate_ids_are_rejected() -> None:
    rows = _parse_rows(SOURCE.read_bytes())
    with pytest.raises(ValueError, match="duplicate ID"):
        _parse_rows(("\n".join(json.dumps(row) for row in [*rows, rows[0]])).encode())


def test_seed_changes_split_assignments(tmp_path: Path) -> None:
    first = prepare_dataset(SOURCE, tmp_path / "first", seed=42)
    second = prepare_dataset(SOURCE, tmp_path / "second", seed=7)
    assert first["splits"]["test"]["ids"] != second["splits"]["test"]["ids"]


def test_exact_duplicates_are_removed_without_changing_original(tmp_path: Path) -> None:
    rows = _parse_rows(SOURCE.read_bytes())
    extra = copy.deepcopy(rows[0])
    extra["id"] = "api-999"
    extra["category"] = "api"
    extra["instruction"] = rows[0]["instruction"].upper()
    extra["response"] = rows[0]["response"]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in [*rows, extra]) + "\n")
    manifest = prepare_dataset(source, tmp_path / "out")
    assert manifest["retained_count"] == 36
    assert manifest["removed_duplicate_ids"] == ["api-999"]


def test_conflicting_duplicate_answers_are_rejected() -> None:
    rows = _parse_rows(SOURCE.read_bytes())
    conflict = dict(rows[0], id="api-999", category="api", response="Different answer")
    with pytest.raises(ValueError, match="conflicting answers"):
        _deduplicate([*rows, conflict])


def test_near_duplicates_are_rejected() -> None:
    rows = _parse_rows(SOURCE.read_bytes())
    duplicate = dict(rows[0], id="api-999", category="api")
    duplicate["instruction"] = rows[0]["instruction"] + " please"
    with pytest.raises(ValueError, match="near-duplicate"):
        _deduplicate([*rows, duplicate])


def test_invalid_input_does_not_create_outputs(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"id": "bad"}\n', encoding="utf-8")
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="expected exactly"):
        prepare_dataset(source, output)
    assert not output.exists()


def test_too_small_category_is_rejected(tmp_path: Path) -> None:
    row = _parse_rows(SOURCE.read_bytes())[0]
    source = tmp_path / "small.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 3"):
        prepare_dataset(source, tmp_path / "processed")
