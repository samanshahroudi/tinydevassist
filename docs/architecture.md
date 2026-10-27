# Architecture and implementation plan

## Objective and boundaries

TinyDevAssist answers narrowly scoped developer-support questions with short, actionable responses. Target topics: Git workflows, Python tracebacks, HTTP status codes, REST APIs, and common backend debugging. Avoid unsupported code execution, private tickets, credentials, vulnerability exploitation, and claims about live systems. Small-model scale is more appropriate for concise, structured responses than broad open-ended software engineering.

## Components

1. **Source records:** Public, permission-compatible technical-support examples or newly written examples. Record provenance, version, license, and redistribution permissions. No unknown-license scraped data.
2. **Preparation:** Implemented deterministic Unicode/whitespace normalization, exact-duplicate removal, near-duplicate rejection, and category-stratified train/validation/test split with seed 42; store generated split manifests and hashes outside Git.
3. **Adaptation:** `HuggingFaceTB/SmolLM2-360M-Instruct` with prompt format from the tokenizer's chat template; LoRA via PEFT and Transformers Trainer; CPU inference and optional CUDA training; configurable hyperparameters. Keep checkpoints outside Git.
4. **Evaluation:** Fixed held-out test cases, paired base and adapted generations, latency and token counts, and text inspection of all six response pairs. Baseline parameter bytes and process RSS snapshots were recorded; CPU peak memory and an independently labeled quality score were not measured. Never evaluate on train examples.
5. **Serving:** Lazy-loaded model, bounded generation controls, Gradio UI on port 7860, Docker non-root user, basic local smoke test, and public Space health/behavior verification.
6. **Delivery:** GitHub Actions lint/tests, dataset and model cards, versioned configuration, reproducibility commands, Docker/Space documentation.

## Deployment architecture

`Hugging Face model repo (weights or adapter + base reference)` → `Hugging Face Docker Space` → `Gradio interface / bounded inference`.

The Space is configured for CPU and loads the PEFT adapter over the pinned upstream base. The local baseline recorded about 1.45 GB of fp32 parameter bytes and roughly 1.8 GB process RSS after individual generations; actual Space memory, startup time, and runtime remain unverified. Repository and Space names must come from confirmed user configuration.

## Risk register

- CPU training may be impractically slow; use available CUDA for a reduced validated training run, explicitly recording hardware/time, or report uncompleted training.
- Curated synthetic-like examples can overfit and lack realistic generalization; use an independent held-out dataset and qualitative errors.
- A 360M model may hallucinate; show limitations and discourage production use without verification.
- Model/license compatibility, package versions, and Hugging Face publishing permissions must be confirmed before publication.

## LoRA training boundary (Milestone 6 implementation)

The training CLI validates the deterministic source and **all** split hashes,
requires a pinned model revision and a measured baseline before any execution,
and uses only `train` and `validation` rows. The tokenizer chat template
constructs sequences; system/user tokens and padding are masked (`-100`) from
the supervised loss. PEFT adapts `q_proj` and `v_proj`, with bounded steps,
no external metric reporting, and adapter-only saves. The `test` split is
**not** used for fitting or hyperparameter selection. A bounded CPU training run
and a paired six-prompt generation comparison have now executed. The
[text inspection](qualitative_review.md) identifies specific errors without
claiming a general quality score. Public serving remains to be verified.
