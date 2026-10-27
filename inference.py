"""Run reproducible baseline generation on held-out developer-support prompts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from tinydevassist.data import sha256_bytes
from tinydevassist.inference import BaselineGenerator


def load_examples(path: Path, limit: int) -> list[dict[str, str]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(
            isinstance(row.get(key), str) and row[key].strip() for key in ("id", "instruction")
        ):
            raise ValueError("each example needs nonempty id and instruction")
        examples.append({"id": row["id"], "instruction": row["instruction"]})
        if len(examples) == limit:
            break
    if not examples:
        raise ValueError("input has no examples")
    return examples


def run_baseline(
    generator: BaselineGenerator,
    examples: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    results = []
    for row in examples:
        result = generator.generate(
            row["instruction"], max_new_tokens=max_new_tokens, temperature=temperature
        )
        results.append({"id": row["id"], **result.to_dict()})
    latencies = [row["metrics"]["latency_seconds"] for row in results]
    return {
        "schema_version": 1,
        "status": "measured",
        "model_id": generator.model_id,
        "requested_revision": generator.requested_revision,
        "resolved_revision": generator.resolved_revision,
        "device": generator.device,
        "python_version": platform.python_version(),
        "torch_version": str(generator.torch.__version__),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "generation": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "summary": {
            "samples": len(results),
            "median_latency_seconds": median(latencies),
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "parameter_count": generator.parameter_count,
            "parameter_bytes": generator.parameter_bytes,
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--revision", default="main", help="Pin to an upstream commit SHA")
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.json"))
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if not args.input.is_file():
        parser.error(f"missing {args.input}; run python prepare_data.py first")
    if args.output.exists():
        print(f"Baseline not completed: refusing to overwrite {args.output}", file=sys.stderr)
        return 1
    examples = load_examples(args.input, args.max_samples)
    try:
        generator = BaselineGenerator(args.model_id, args.revision, args.device)
        report = run_baseline(
            generator,
            examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Baseline not completed: {exc}", file=sys.stderr)
        return 1
    report["input_sha256"] = sha256_bytes(args.input.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Measured {len(report['results'])} samples; saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
