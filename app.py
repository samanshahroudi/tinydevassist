"""CPU Gradio demo for the pinned TinyDevAssist LoRA adapter."""

from __future__ import annotations

import os
from functools import lru_cache

from tinydevassist.evaluation import AdapterGenerator

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
BASE_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
EXAMPLES = [
    "How can I inspect what changed before committing in Git?",
    "What should I check first when a Python import fails?",
    "How should an API respond when a requested resource is missing?",
    "What can I inspect when an HTTP request times out?",
]


@lru_cache(maxsize=1)
def load_generator() -> AdapterGenerator:
    adapter = os.environ.get("HF_MODEL_REPO") or os.environ.get("TINYDEVASSIST_ADAPTER_PATH")
    if not adapter:
        raise RuntimeError("Set HF_MODEL_REPO or TINYDEVASSIST_ADAPTER_PATH")
    return AdapterGenerator(BASE_MODEL_ID, BASE_REVISION, adapter, device="cpu")


def respond(prompt: str, temperature: float, max_new_tokens: int) -> str:
    """Validate UI inputs before loading a potentially large model."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a developer-support question")
    if len(prompt) > 2000:
        raise ValueError("Keep the question under 2,000 characters")
    if not isinstance(max_new_tokens, int) or not 16 <= max_new_tokens <= 256:
        raise ValueError("Maximum tokens must be between 16 and 256")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    return (
        load_generator()
        .generate(prompt.strip(), max_new_tokens=max_new_tokens, temperature=float(temperature))
        .response
    )


def create_demo():
    import gradio as gr

    with gr.Blocks(title="TinyDevAssist") as demo:
        gr.Markdown(
            "# TinyDevAssist\n"
            "A small developer-support model adapted with six LoRA training steps. "
            "Its answers may be wrong; check commands before running them. "
            "Observed errors include fabricated API calls and incorrect Git commands. "
            "The six-prompt comparison does not establish better quality than the base model."
        )
        prompt = gr.Textbox(label="Developer question", lines=3, max_lines=5)
        with gr.Row():
            temperature = gr.Slider(0, 1, value=0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(16, 256, value=96, step=1, label="Maximum new tokens")
        answer = gr.Textbox(label="Response", lines=8)
        gr.Button("Generate").click(
            respond, inputs=[prompt, temperature, max_tokens], outputs=answer, api_name="respond"
        )
        gr.Examples(EXAMPLES, inputs=prompt)
        gr.Markdown(
            f"Base: `{BASE_MODEL_ID}` at `{BASE_REVISION}`. "
            "The adapter is served from the configured Hugging Face model repository."
        )
    return demo


if __name__ == "__main__":
    create_demo().launch(server_name="0.0.0.0", server_port=7860)
