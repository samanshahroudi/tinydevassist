"""Keep notebooks valid and honest without requiring notebook extras in CI."""

from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_valid_json_and_have_code_and_explanation():
    root = Path(__file__).resolve().parents[1]
    for name in ("01_dataset_exploration.ipynb", "02_baseline_model.ipynb"):
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
        assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])


def test_baseline_notebook_requires_opt_in_to_download_model():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/02_baseline_model.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "RUN_BASELINE = False" in code
