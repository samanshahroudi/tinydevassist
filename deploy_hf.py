"""Publish the verified adapter and Docker Space from an authenticated local shell."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)
SPACE_FILES = {
    "README.md": ROOT / "space/README.md",
    "Dockerfile": ROOT / "space/Dockerfile",
    "requirements.txt": ROOT / "space/requirements.txt",
    "app.py": ROOT / "app.py",
    "src/tinydevassist/__init__.py": ROOT / "src/tinydevassist/__init__.py",
    "src/tinydevassist/data.py": ROOT / "src/tinydevassist/data.py",
    "src/tinydevassist/evaluation.py": ROOT / "src/tinydevassist/evaluation.py",
}


def deployment_manifest(adapter_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    model_files = {"README.md": ROOT / "space/MODEL_CARD.md"}
    model_files.update({name: adapter_dir / name for name in ADAPTER_FILES})
    for path in (*model_files.values(), *SPACE_FILES.values()):
        if not path.is_file():
            raise ValueError(f"required deployment file is missing: {path}")
    return model_files, SPACE_FILES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO"))
    parser.add_argument("--space-repo", default=os.environ.get("HF_SPACE_REPO"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/tinydevassist-lora")
    parser.add_argument("--execute", action="store_true", help="Create public Hub repositories")
    args = parser.parse_args(argv)
    try:
        if not args.model_repo or not REPO_ID.fullmatch(args.model_repo):
            raise ValueError("provide an exact namespace/model repository ID")
        if not args.space_repo or not REPO_ID.fullmatch(args.space_repo):
            raise ValueError("provide an exact namespace/Space repository ID")
        model_files, space_files = deployment_manifest(args.adapter)
        print(f"Model repository: {args.model_repo} ({len(model_files)} curated files)")
        print(f"Docker Space: {args.space_repo} ({len(space_files)} curated files)")
        if not args.execute:
            print("Dry run only. Add --execute after logging in with `hf auth login`.")
            return 0

        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        api.whoami()  # Fail before creating any repository if no valid credential exists.
        api.create_repo(args.model_repo, repo_type="model", exist_ok=False)
        api.create_commit(
            repo_id=args.model_repo,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in model_files.items()
            ],
            commit_message="Publish TinyDevAssist LoRA adapter and model card",
        )
        api.create_repo(args.space_repo, repo_type="space", space_sdk="docker", exist_ok=False)
        api.add_space_variable(args.space_repo, key="HF_MODEL_REPO", value=args.model_repo)
        api.create_commit(
            repo_id=args.space_repo,
            repo_type="space",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in space_files.items()
            ],
            commit_message="Publish TinyDevAssist CPU demo",
        )
        print(f"Uploaded model: https://huggingface.co/{args.model_repo}")
        print(f"Uploaded Space: https://huggingface.co/spaces/{args.space_repo}")
        print("Verify the Space build status and run a public inference smoke test next.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Deployment incomplete: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
