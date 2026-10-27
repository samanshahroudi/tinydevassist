# Model selection and upstream verification

**Selected candidate:** [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

## Verified from the upstream model card (2026-10-27)

- The official model page identifies the license as **Apache-2.0**, the language as primarily **English**, and the architecture as a decoder-only model in the SmolLM2 family.
- The official page shows an instruction-tuned 360M-class model with a Transformers example using `AutoTokenizer`, `AutoModelForCausalLM`, and `tokenizer.apply_chat_template`.
- The card explicitly warns that answers may be inaccurate or biased. We will require human verification for important commands and avoid production-use claims.
- Upstream reference: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct ; research paper: https://arxiv.org/abs/2502.02737 .

## Alternatives and rationale

- **TinyLlama 1.1B:** more capacity, but greater CPU memory, latency, and training requirements.
- **SmolLM2-135M-Instruct:** smaller footprint, but less capacity for even short support answers.
- **SmolLM2-360M-Instruct:** reasonable size to attempt CPU inference and laptop-scale LoRA fine-tuning, without claiming measured feasibility.

## Configuration and reproducibility

`configs/model.json` records the candidate and 512-token experiment length. `revision: main` is **not immutable**; before training, resolve and record the upstream commit SHA, tokenizer files, exact dependency versions, and local hardware. Do not claim pinned reproducibility until that is done.

## Not yet verified

The weights and tokenizer have **not** been downloaded or executed in this workspace. GPU compatibility, CPU inference latency, peak memory, quantization support, and LoRA training success are unknown. The upstream license is verified from metadata; package compatibility and runtime behavior remain future milestones.
