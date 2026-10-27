"""ZeroGPU Gradio entry point; model placement happens before requests arrive."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import core_app  # noqa: E402

from tinydevassist.evaluation import AdapterGenerator  # noqa: E402

adapter_id = os.environ.get("HF_MODEL_REPO")
if not adapter_id:
    raise RuntimeError("Set the HF_MODEL_REPO Space variable to the public adapter ID")

# ZeroGPU intercepts this startup placement and restores the weights for GPU calls.
generator = AdapterGenerator(core_app.BASE_MODEL_ID, core_app.BASE_REVISION, adapter_id, "cuda")


@spaces.GPU(duration=60)
def respond(prompt: str, temperature: float, max_new_tokens: int) -> str:
    return core_app.respond(prompt, temperature, max_new_tokens, generator=generator)


demo = core_app.create_demo(responder=respond)

if __name__ == "__main__":
    demo.launch()
