"""Upload the verified adapter demo to an existing Gradio ZeroGPU Space."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from deploy_hf import REPO_ID, deployment_manifest, verify_matching_model_files

ROOT = Path(__file__).resolve().parent
SPACE_FILES = {
    "README.md": ROOT / "space/zerogpu/README.md",
    "requirements.txt": ROOT / "space/zerogpu/requirements.txt",
    "app.py": ROOT / "space/zerogpu/app.py",
    "core_app.py": ROOT / "app.py",
    "src/tinydevassist/__init__.py": ROOT / "src/tinydevassist/__init__.py",
    "src/tinydevassist/data.py": ROOT / "src/tinydevassist/data.py",
    "src/tinydevassist/evaluation.py": ROOT / "src/tinydevassist/evaluation.py",
}


def zerogpu_manifest() -> dict[str, Path]:
    for path in SPACE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required ZeroGPU file is missing: {path}")
    return SPACE_FILES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO"))
    parser.add_argument("--space-repo", default=os.environ.get("HF_SPACE_REPO"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/tinydevassist-lora")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.model_repo or not REPO_ID.fullmatch(args.model_repo):
            raise ValueError("provide an exact namespace/model repository ID")
        if not args.space_repo or not REPO_ID.fullmatch(args.space_repo):
            raise ValueError("provide an exact namespace/Space repository ID")
        model_files, _ = deployment_manifest(args.adapter)
        space_files = zerogpu_manifest()
        print(f"Existing model: {args.model_repo} ({len(model_files)} curated files)")
        print(f"ZeroGPU Space: {args.space_repo} ({len(space_files)} curated files)")
        if not args.execute:
            print("Dry run only. Add --execute to verify the model and upload the Space package.")
            return 0

        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        api = HfApi()
        api.whoami()
        model_info = api.model_info(args.model_repo)
        if model_info.private or not set(model_files).issubset(
            {file.rfilename for file in model_info.siblings}
        ):
            raise ValueError("the public model is missing curated adapter files")
        verify_matching_model_files(
            model_files, lambda name: Path(hf_hub_download(args.model_repo, name, token=False))
        )

        space_info = api.space_info(args.space_repo)
        runtime = api.get_space_runtime(args.space_repo)
        if space_info.sdk != "gradio" or runtime.requested_hardware != "zero-a10g":
            raise ValueError("the existing Space must be a Gradio ZeroGPU Space")
        api.add_space_variable(args.space_repo, key="HF_MODEL_REPO", value=args.model_repo)
        api.create_commit(
            repo_id=args.space_repo,
            repo_type="space",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in space_files.items()
            ],
            commit_message="Publish TinyDevAssist ZeroGPU demo",
        )
        print(f"Uploaded Space: https://huggingface.co/spaces/{args.space_repo}")
        print("Verify the Space build and public inference before claiming deployment.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ZeroGPU upload incomplete: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
