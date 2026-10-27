---
language: en
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - developer-support
---

# TinyDevAssist LoRA adapter

This is a PEFT LoRA adapter for
`HuggingFaceTB/SmolLM2-360M-Instruct` at upstream revision
`a10cc1512eabd3dde888204e902eca88bddb4951`. It is an educational
developer-support experiment, not a production assistant.

The project-authored Apache-2.0 seed contains 36 English question-answer pairs
across Git, Python, HTTP, APIs, debugging, and operations. A deterministic
seed-42 split provided 24 train, 6 validation, and 6 test records. Training
used one epoch and six optimizer steps with rank 8, alpha 16, dropout 0.05,
and `q_proj`/`v_proj` targets. The CPU run used Python 3.11.16, PyTorch
2.14.1+cpu, Transformers 4.57.6, and PEFT 0.21.2. Training loss was 3.27303;
validation loss was 3.39398.

A paired generation comparison on the six held-out prompts produced nonempty
responses from both base and adapter. Mean generation latency on that CPU run
was 9.520 seconds for the base and 10.301 seconds for the adapter. No
independent quality score or improvement claim is available. The small authored
dataset cannot establish generalization. Responses may contain wrong or unsafe
commands; verify them before use.

Load this adapter with PEFT on the pinned base revision. See the
[source repository](https://github.com/samanshahroudi/tinydevassist) for the
data card, scripts, configuration, tests, and evaluation protocol.
