"""Non-invasive diagnostics for local development environments."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from typing import Any

from tinydevassist import DEFAULT_BASE_MODEL, __version__

OPTIONAL_MODULES = ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "gradio")
RECOMMENDED_PYTHON = ((3, 11), (3, 12))


def collect_environment() -> dict[str, Any]:
    """Report dependency presence without loading models or making network requests."""
    version = sys.version_info[:2]
    return {
        "project_version": __version__,
        "base_model_candidate": DEFAULT_BASE_MODEL,
        "python": platform.python_version(),
        "python_recommended_for_ml": version in RECOMMENDED_PYTHON,
        "platform": platform.platform(),
        "optional_dependencies_present": {
            name: importlib.util.find_spec(name) is not None for name in OPTIONAL_MODULES
        },
        "note": "Presence does not verify package compatibility, GPU support, or model access.",
    }


def environment_json() -> str:
    """Return stable machine-readable diagnostics."""
    return json.dumps(collect_environment(), indent=2, sort_keys=True)
