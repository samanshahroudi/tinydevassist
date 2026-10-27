"""Check a running local or public TinyDevAssist Space and generate one answer."""

from __future__ import annotations

import argparse
import sys
from urllib.request import urlopen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args(argv)
    try:
        with urlopen(args.url, timeout=30) as response:  # noqa: S310 - explicit user URL
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
        from gradio_client import Client

        answer = Client(args.url, verbose=False).predict(
            "How can I check my Git working tree before committing?", 0, 64, api_name="/respond"
        )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("demo returned an empty answer")
        print("HTTP 200; inference returned a nonempty response")
        return 0
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
