"""Development commands for the TinyDevAssist package."""

from __future__ import annotations

import argparse

from tinydevassist.environment import environment_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TinyDevAssist developer utilities")
    parser.add_argument("command", choices=("doctor",), help="Run environment diagnostics")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(environment_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
