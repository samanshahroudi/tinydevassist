"""Compare the measured base model with the local LoRA adapter on held-out prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference import load_examples

from tinydevassist.evaluation import AdapterGenerator, comparison_report
from tinydevassist.training import PINNED_REVISION, load_training_config, load_verified_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--source", type=Path, default=Path("data/samples/developer_support_v1.jsonl")
    )
    parser.add_argument("--adapter", type=Path, default=Path("checkpoints/tinydevassist-lora"))
    parser.add_argument("--baseline-report", type=Path, default=Path("outputs/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if not PINNED_REVISION.fullmatch(args.revision):
            raise ValueError("--revision must be a 40-character model commit SHA")
        if args.output.exists():
            raise ValueError(f"refusing to overwrite existing evaluation report: {args.output}")
        config = load_training_config(args.config, args.revision)
        manifest, _ = load_verified_splits(args.processed_dir, args.source, config)
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        expected_hash = manifest["splits"]["test"]["sha256"]
        if (
            baseline.get("input_sha256") != expected_hash
            or baseline.get("resolved_revision") != args.revision
        ):
            raise ValueError("baseline does not match this held-out split and revision")
        base_by_id = {row["id"]: row for row in baseline.get("results", [])}
        examples = load_examples(args.processed_dir / "test.jsonl", limit=10_000)
        if set(base_by_id) != {row["id"] for row in examples}:
            raise ValueError("baseline must cover exactly the held-out prompts")
        generator = AdapterGenerator(
            config.base_model_id, args.revision, str(args.adapter), args.device
        )
        rows = [
            {
                "id": row["id"],
                "base": {
                    "response": base_by_id[row["id"]]["response"],
                    **base_by_id[row["id"]]["metrics"],
                },
                "adapter": generator.generate(row["instruction"]).to_dict(),
            }
            for row in examples
        ]
        report = comparison_report(
            rows, model_id=config.base_model_id, revision=args.revision, test_sha256=expected_hash
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Compared {len(rows)} held-out prompts; saved {args.output}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Evaluation not completed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
