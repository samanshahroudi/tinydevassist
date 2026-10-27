"""Validated, bounded LoRA SFT training; heavyweight imports occur only on execution."""

from __future__ import annotations

import json
import math
import platform
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinydevassist.data import SPLIT_NAMES, sha256_bytes, training_messages

PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class LoRASettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationSettings:
    epochs: int
    learning_rate: float
    train_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    max_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    base_model_id: str
    revision: str
    seed: int
    max_sequence_length: int
    lora: LoRASettings
    optimization: OptimizationSettings


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where}: expected exactly {sorted(expected)}")
    return value


def _int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return float(value)


def load_training_config(path: Path, revision: str | None = None) -> TrainingConfig:
    """Parse a strict JSON config without importing torch or downloading model weights."""
    raw = _exact_keys(
        json.loads(path.read_text(encoding="utf-8")),
        {"base_model_id", "revision", "seed", "max_sequence_length", "lora", "optimization"},
        "config",
    )
    lora = _exact_keys(raw["lora"], {"rank", "alpha", "dropout", "target_modules"}, "lora")
    opt = _exact_keys(
        raw["optimization"],
        {
            "epochs",
            "learning_rate",
            "train_batch_size",
            "gradient_accumulation_steps",
            "weight_decay",
            "max_steps",
        },
        "optimization",
    )
    modules = lora["target_modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(m, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", m) for m in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("lora.target_modules must contain unique module names")
    model_id = raw["base_model_id"]
    model_revision = revision if revision is not None else raw["revision"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("base_model_id must be nonempty")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("revision must be nonempty")
    return TrainingConfig(
        base_model_id=model_id,
        revision=model_revision,
        seed=_int(raw["seed"], "seed", 0, 2**32 - 1),
        max_sequence_length=_int(raw["max_sequence_length"], "max_sequence_length", 64, 8192),
        lora=LoRASettings(
            rank=_int(lora["rank"], "lora.rank", 1, 256),
            alpha=_int(lora["alpha"], "lora.alpha", 1, 1024),
            dropout=_number(lora["dropout"], "lora.dropout", 0, 0.5),
            target_modules=tuple(modules),
        ),
        optimization=OptimizationSettings(
            epochs=_int(opt["epochs"], "optimization.epochs", 1, 10),
            learning_rate=_number(opt["learning_rate"], "optimization.learning_rate", 1e-7, 1e-2),
            train_batch_size=_int(opt["train_batch_size"], "optimization.train_batch_size", 1, 32),
            gradient_accumulation_steps=_int(
                opt["gradient_accumulation_steps"],
                "optimization.gradient_accumulation_steps",
                1,
                128,
            ),
            weight_decay=_number(opt["weight_decay"], "optimization.weight_decay", 0, 1),
            max_steps=_int(opt["max_steps"], "optimization.max_steps", 1, 1000),
        ),
    )


def _read_split(path: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != spec["sha256"]:
        raise ValueError(f"{path}: SHA-256 differs from manifest; rerun prepare_data.py")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != spec["count"] or [r.get("id") for r in rows] != spec["ids"]:
        raise ValueError(f"{path}: count or ordered IDs differ from manifest")
    if dict(sorted(Counter(r.get("category") for r in rows).items())) != spec["categories"]:
        raise ValueError(f"{path}: categories differ from manifest")
    for row in rows:
        if not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("id", "instruction", "response")
        ):
            raise ValueError(f"{path}: invalid record fields")
        if row.get("messages") != training_messages(row["instruction"], row["response"]):
            raise ValueError(f"{path}: messages do not match source fields")
    return rows


def load_verified_splits(
    processed_dir: Path, source_path: Path, config: TrainingConfig
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify source and all split hashes, including held-out test, without training on test."""
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("seed") != config.seed:
        raise ValueError("manifest schema or seed differs from training config")
    if manifest.get("source_path") != source_path.name or manifest.get(
        "source_sha256"
    ) != sha256_bytes(source_path.read_bytes()):
        raise ValueError("source dataset hash differs from manifest")
    specs = manifest.get("splits")
    if not isinstance(specs, dict) or set(specs) != set(SPLIT_NAMES):
        raise ValueError("manifest must contain train, validation and test splits")
    splits = {
        name: _read_split(processed_dir / f"{name}.jsonl", specs[name]) for name in SPLIT_NAMES
    }
    ids = [row["id"] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ID leakage between splits")
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("all three splits must be nonempty")
    return manifest, splits


def training_plan(config: TrainingConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """A plan is NOT a training result; it contains only checked input metadata."""
    count = manifest["splits"]["train"]["count"]
    opt = config.optimization
    return {
        "status": "dry_run_only",
        "training_executed": False,
        "config": asdict(config),
        "dataset_source_sha256": manifest["source_sha256"],
        "split_sha256": {name: manifest["splits"][name]["sha256"] for name in SPLIT_NAMES},
        "split_counts": {name: manifest["splits"][name]["count"] for name in SPLIT_NAMES},
        "estimated_optimizer_steps": min(
            opt.max_steps,
            math.ceil(math.ceil(count / opt.train_batch_size) / opt.gradient_accumulation_steps)
            * opt.epochs,
        ),
        "requires_pinned_revision": not bool(PINNED_REVISION.fullmatch(config.revision)),
        "requires_measured_baseline": True,
    }


def verify_baseline(path: Path, config: TrainingConfig, manifest: dict[str, Any]) -> None:
    """Require a measured baseline on exactly this model/revision/held-out test split."""
    if not PINNED_REVISION.fullmatch(config.revision):
        raise ValueError("execution requires --revision with a 40-character model commit SHA")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "measured"
        or baseline.get("model_id") != config.base_model_id
        or baseline.get("requested_revision") != config.revision
        or baseline.get("resolved_revision") != config.revision
        or baseline.get("input_sha256") != manifest["splits"]["test"]["sha256"]
    ):
        raise ValueError("baseline must be measured with the pinned model and matching test hash")
    expected_ids = set(manifest["splits"]["test"]["ids"])
    actual_results = baseline.get("results")
    if (
        not isinstance(actual_results, list)
        or any(not isinstance(r, dict) for r in actual_results)
        or {r.get("id") for r in actual_results} != expected_ids
        or len(actual_results) != len(expected_ids)
    ):
        raise ValueError("baseline must cover every held-out test example")
    for result in actual_results:
        metrics = result.get("metrics")
        if (
            not isinstance(result.get("response"), str)
            or not isinstance(metrics, dict)
            or type(metrics.get("latency_seconds")) not in (int, float)
            or not math.isfinite(metrics["latency_seconds"])
            or metrics["latency_seconds"] < 0
            or type(metrics.get("generated_tokens")) is not int
            or metrics["generated_tokens"] < 0
        ):
            raise ValueError("baseline report lacks measured generation metrics")


def tokenize_supervised_example(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> dict[str, list[int]]:
    """Use the upstream chat template and mask system/user tokens from the SFT loss."""
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant messages")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(prompt, list) or not isinstance(full, list) or full[: len(prompt)] != prompt:
        raise ValueError("chat-template prompt must be a prefix of the completed conversation")
    if len(full) > max_length or len(prompt) >= len(full):
        raise ValueError("sample exceeds max_sequence_length or has no assistant target tokens")
    if any(type(token) is not int or token < 0 for token in full):
        raise ValueError("chat template returned invalid token IDs")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt) + full[len(prompt) :],
    }


def pad_supervised_batch(
    features: list[dict[str, list[int]]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Right-pad labels with -100 so padding never contributes to training loss."""
    if not features or type(pad_token_id) is not int:
        raise ValueError("expected nonempty features and integer pad token")
    max_len = max(len(row["input_ids"]) for row in features)
    result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in features:
        n = len(row["input_ids"])
        if not n or len(row["labels"]) != n or len(row["attention_mask"]) != n:
            raise ValueError("inconsistent tokenized feature lengths")
        padding = max_len - n
        result["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(row["attention_mask"] + [0] * padding)
        result["labels"].append(row["labels"] + [-100] * padding)
    return result


def run_training(
    config: TrainingConfig,
    manifest: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Train a PEFT adapter using Transformers Trainer; never touch the test split."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    try:
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("install training dependencies with pip install -e '.[train]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id, revision=config.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = {
        name: [
            tokenize_supervised_example(tokenizer, row["messages"], config.max_sequence_length)
            for row in splits[name]
        ]
        for name in ("train", "validation")
    }
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        revision=config.revision,
        trust_remote_code=False,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is not None and resolved != config.revision:
        raise ValueError("loaded model revision does not match pinned commit")
    lora = config.lora
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0 or trainable_params >= total_params:
        raise RuntimeError("LoRA parameter freezing did not produce a partial trainable model")
    opt = config.optimization
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=opt.epochs,
        max_steps=opt.max_steps,
        learning_rate=opt.learning_rate,
        per_device_train_batch_size=opt.train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        weight_decay=opt.weight_decay,
        seed=config.seed,
        data_seed=config.seed,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        disable_tqdm=True,
    )

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        padded = pad_supervised_batch(features, tokenizer.pad_token_id)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in padded.items()}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=collate,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))  # PEFT saves adapters, not merged base weights.
    tokenizer.save_pretrained(str(output_dir))
    return {
        "schema_version": 1,
        "status": "trained",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "base_model_id": config.base_model_id,
        "requested_revision": config.revision,
        "resolved_revision": resolved,
        "config": asdict(config),
        "source_sha256": manifest["source_sha256"],
        "split_sha256": {name: manifest["splits"][name]["sha256"] for name in SPLIT_NAMES},
        "train_count": len(encoded["train"]),
        "validation_count": len(encoded["validation"]),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_metrics": train_result.metrics,
        "validation_metrics": eval_metrics,
        "adapter_path": str(output_dir),
    }
