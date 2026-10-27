"""Fast, offline audit of the tracked illustrative seed records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data/samples/developer_support_v1.jsonl"
CATEGORIES = {"git", "python", "http", "api", "debugging", "ops"}
REQUIRED_FIELDS = {"id", "category", "instruction", "response", "source"}


def test_seed_dataset_schema_and_provenance() -> None:
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 36
    assert Counter(row["category"] for row in rows) == {category: 6 for category in CATEGORIES}
    assert all(set(row) == REQUIRED_FIELDS for row in rows)
    assert all(row["source"] == "project-authored-v1" for row in rows)
    assert all(row["id"].startswith(f"{row['category']}-") for row in rows)
    assert all(len(row["instruction"].strip()) > 12 for row in rows)
    assert all(len(row["response"].strip()) > 30 for row in rows)


def test_seed_dataset_has_unique_ids_and_questions() -> None:
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({" ".join(row["instruction"].casefold().split()) for row in rows}) == len(rows)
