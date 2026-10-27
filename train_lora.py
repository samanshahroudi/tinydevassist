"""Plan or explicitly execute a bounded LoRA fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinydevassist.training import (
    PINNED_REVISION,
    load_training_config,
    load_verified_splits,
    run_training,
    training_plan,
    verify_baseline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--source", type=Path, default=Path("data/samples/developer_support_v1.jsonl")
    )
    parser.add_argument("--baseline-report", type=Path, default=Path("outputs/baseline.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/tinydevassist-lora"))
    parser.add_argument("--report", type=Path, default=Path("outputs/train_lora.json"))
    parser.add_argument("--revision", help="Immutable 40-character model commit SHA")
    parser.add_argument(
        "--execute", action="store_true", help="Opt in to model download and training"
    )
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config, args.revision)
        manifest, splits = load_verified_splits(args.processed_dir, args.source, config)
        if not args.execute:
            print(json.dumps(training_plan(config, manifest), indent=2))
            return 0
        if not PINNED_REVISION.fullmatch(config.revision):
            raise ValueError("--execute requires a pinned 40-character --revision")
        verify_baseline(args.baseline_report, config, manifest)
        if args.report.exists():
            raise ValueError(f"refusing to overwrite existing training report: {args.report}")
        result = run_training(config, manifest, splits, args.output_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Training completed; adapter: {args.output_dir}; report: {args.report}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"LoRA training not completed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
