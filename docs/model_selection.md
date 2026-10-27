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
- **SmolLM2-360M-Instruct:** the selected small model; CPU inference and a six-step LoRA run have since completed.

## Configuration and reproducibility

`configs/model.json` retains `revision: main` as a convenience default, which
is **not immutable**. The measured runs explicitly passed upstream commit
`a10cc1512eabd3dde888204e902eca88bddb4951`; the reports record Python,
PyTorch, Transformers, and input hashes. Deployment uses the same revision.

## Current runtime boundary

The pinned revision `a10cc1512eabd3dde888204e902eca88bddb4951` and tokenizer
were downloaded and exercised for CPU baseline inference, bounded LoRA training,
and paired adapter generation. The six-prompt latency figures and validation
loss are recorded in [`evaluation.md`](evaluation.md) and
[`execution.md`](execution.md). Peak CPU memory, quantization, independent
response quality, and sustained hosting reliability remain unverified. A
Gradio ZeroGPU Space built and one anonymous hosted inference returned a
nonempty answer; that verifies a narrow GPU serving path, not general
compatibility or answer quality. See the [deployment record](deployment.md).
