---
title: TinyDevAssist
emoji: 🛠️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12.12
app_file: app.py
suggested_hardware: zero-a10g
short_description: LoRA developer support on ZeroGPU
---

# TinyDevAssist

This Gradio Space runs the public TinyDevAssist PEFT adapter on ZeroGPU. The
base model is pinned to an upstream commit and the adapter repository is set
through the public `HF_MODEL_REPO` Space variable. Requests are limited to a
512-token prompt-plus-completion budget.

The 36-example authored training seed and six held-out prompts are too small
to establish generalization. Observed failures include fabricated API usage
and misleading Git advice. Check every suggested command before use. See the
[source repository](https://github.com/samanshahroudi/tinydevassist) and
[model card](https://huggingface.co/saman-shahroudi/tinydevassist) for the
experiment, data provenance, and limitations.
