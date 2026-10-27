"""Deterministic, dependency-free preparation of developer-support SFT examples."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are TinyDevAssist, a concise developer-support assistant. "
    "Give actionable, cautious answers. If uncertain, say so."
)
REQUIRED_FIELDS = frozenset({"id", "category", "instruction", "response", "source"})
SPLIT_NAMES = ("train", "validation", "test")
NEAR_DUPLICATE_THRESHOLD = 0.92


def normalize_text(value: str) -> str:
    """Apply Unicode NFC and collapse whitespace without changing technical casing."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def question_key(value: str) -> str:
    """Canonical key for duplicate checks; punctuation is intentionally preserved."""
    return normalize_text(value).casefold()


def prompt_messages(instruction: str) -> list[dict[str, str]]:
    """Return role/content messages; tokenizer applies the actual model chat template."""
    cleaned = normalize_text(instruction)
    if not cleaned:
        raise ValueError("instruction must not be blank")
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": cleaned},
    ]


def training_messages(instruction: str, response: str) -> list[dict[str, str]]:
    """Construct SFT messages with the target response as the final assistant turn."""
    answer = normalize_text(response)
    if not answer:
        raise ValueError("response must not be blank")
    return [*prompt_messages(instruction), {"role": "assistant", "content": answer}]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_rows(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source must be UTF-8") from exc
    rows = []
    ids = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or set(row) != REQUIRED_FIELDS:
            raise ValueError(f"line {line_number}: expected exactly {sorted(REQUIRED_FIELDS)}")
        if not all(isinstance(value, str) for value in row.values()):
            raise ValueError(f"line {line_number}: all fields must be strings")
        cleaned = {key: normalize_text(value) for key, value in row.items()}
        if any(not value for value in cleaned.values()):
            raise ValueError(f"line {line_number}: empty field")
        if cleaned["id"] in ids:
            raise ValueError(f"line {line_number}: duplicate ID {cleaned['id']}")
        if not cleaned["id"].startswith(f"{cleaned['category']}-"):
            raise ValueError(f"line {line_number}: ID/category mismatch")
        ids.add(cleaned["id"])
        rows.append(cleaned)
    if not rows:
        raise ValueError("source contains no examples")
    return sorted(rows, key=lambda row: row["id"])


def _deduplicate(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    unique: dict[str, dict[str, str]] = {}
    removed: list[str] = []
    for row in rows:
        key = question_key(row["instruction"])
        previous = unique.get(key)
        if previous is not None:
            if previous["response"] != row["response"]:
                raise ValueError(
                    f"conflicting answers for duplicate question: {previous['id']} / {row['id']}"
                )
            removed.append(row["id"])
            continue
        unique[key] = row
    retained = list(unique.values())
    # Review near-duplicates instead of silently discarding potentially different answers.
    for index, first in enumerate(retained):
        for second in retained[index + 1 :]:
            ratio = SequenceMatcher(
                None, question_key(first["instruction"]), question_key(second["instruction"])
            ).ratio()
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                raise ValueError(
                    f"near-duplicate questions need review: {first['id']} / {second['id']} "
                    f"(similarity {ratio:.3f})"
                )
    return retained, removed


def _split_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    by_category: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    splits: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    for category, group in sorted(by_category.items()):
        if len(group) < 3:
            raise ValueError(f"category {category!r} needs at least 3 distinct records")
        # A per-category derived seed prevents other categories from changing this split.
        digest = hashlib.sha256(f"{seed}:{category}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        shuffled = sorted(group, key=lambda row: row["id"])
        rng.shuffle(shuffled)
        n_holdout = max(1, round(len(shuffled) * 0.15))
        if 2 * n_holdout >= len(shuffled):
            n_holdout = 1
        splits["validation"].extend(shuffled[:n_holdout])
        splits["test"].extend(shuffled[n_holdout : 2 * n_holdout])
        splits["train"].extend(shuffled[2 * n_holdout :])
    return {
        name: sorted(group, key=lambda row: row["id"]) for name, group in splits.items()
    }


def _encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode("utf-8")


def prepare_dataset(input_path: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Validate, deduplicate and split source data, then emit JSONL plus a hash manifest.

    Validation completes before writing outputs. Generated files must remain ignored by Git.
    """
    raw = input_path.read_bytes()
    parsed = _parse_rows(raw)
    rows, removed = _deduplicate(parsed)
    splits = _split_rows(rows, seed)
    all_ids = [row["id"] for group in splits.values() for row in group]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("record leaked across splits")
    all_keys = [question_key(row["instruction"]) for group in splits.values() for row in group]
    if len(all_keys) != len(set(all_keys)):
        raise AssertionError("question leaked across splits")

    output_data: dict[str, bytes] = {}
    for name, group in splits.items():
        formatted = [
            {
                **row,
                "messages": training_messages(row["instruction"], row["response"]),
            }
            for row in group
        ]
        output_data[f"{name}.jsonl"] = _encode_jsonl(formatted)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_path": input_path.name,
        "source_sha256": sha256_bytes(raw),
        "seed": seed,
        "normalization": "Unicode NFC, collapsed whitespace; casefold for duplicate checks",
        "duplicate_policy": (
            "drop exact repeated questions with identical answers; reject conflicts"
        ),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "prompt_format": "role/content messages; apply tokenizer chat template at training time",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_count": len(parsed),
        "retained_count": len(rows),
        "removed_duplicate_ids": removed,
        "splits": {
            name: {
                "count": len(group),
                "categories": dict(sorted(Counter(row["category"] for row in group).items())),
                "ids": [row["id"] for row in group],
                "sha256": sha256_bytes(output_data[f"{name}.jsonl"]),
            }
            for name, group in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in output_data.items():
        (output_dir / filename).write_bytes(content)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
