"""CLI entry point for deterministic TinyDevAssist dataset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinydevassist.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples/developer_support_v1.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_dataset(args.input, args.output_dir, seed=args.seed)
    print(json.dumps({"seed": result["seed"], "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
