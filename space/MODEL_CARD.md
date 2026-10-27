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
Text inspection of the six held-out responses found a fabricated `psutil` API,
misleading Git guidance, and premature operations advice. See the
[review](https://github.com/samanshahroudi/tinydevassist/blob/main/docs/qualitative_review.md)
for the exact cases. These observations do not establish a population error rate.

## Reproduce and use

Clone the [source repository](https://github.com/samanshahroudi/tinydevassist),
install Python 3.11 dependencies with `python -m pip install -e '.[app,dev]'`,
and set `HF_MODEL_REPO` to the exact published adapter repository ID before
running `python app.py`. The demo loads the base at the pinned revision above,
applies this PEFT adapter, and serves Gradio on port 7860. Its tokenized prompt
plus requested new tokens must fit the 512-token training sequence budget.

For the reported experiment, regenerate the seed-42 split with
`python prepare_data.py --seed 42`. The source SHA-256 is
`92ffc379ffe09d71052d9a7cfc9c9b4fe740c540cb7955e4347e2f646bef151f`;
the held-out split SHA-256 is
`706756348bd549ee3db4f453189955b84950760a723751fd287b37134566dde7`.
The [evaluation protocol](https://github.com/samanshahroudi/tinydevassist/blob/main/docs/evaluation.md)
explains how to rerun both model paths and inspect the six response pairs.

Load this adapter with PEFT on the pinned base revision. See the
[source repository](https://github.com/samanshahroudi/tinydevassist) for the
data card, scripts, configuration, tests, and evaluation protocol.
