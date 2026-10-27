# TinyDevAssist — tiny developer-support language model

A reproducible small-language-model engineering portfolio project. **Status: architecture initialized; training, evaluation, and deployment have not run.**

## Goal
Adapt `HuggingFaceTB/SmolLM2-360M-Instruct` using LoRA for concise answers to developer-support questions about Git, Python, HTTP APIs, and backend debugging. Compare it with the unchanged base model, then publish a model card and CPU-friendly Gradio Docker Space.

## Architecture

```mermaid
flowchart LR
  A[Documented public/original data] --> B[Deterministic sanitize + split]
  B --> C[Training/validation/test]
  C --> D[LoRA fine-tuning]
  C --> E[Baseline evaluation]
  D --> F[Adapter / merged model]
  F --> G[Fine-tuned evaluation]
  F --> H[Gradio Docker Space]
  E --> I[Comparison report]
  G --> I
```

See [architecture](docs/architecture.md), [decisions](docs/decisions.md), and [execution milestones](docs/execution.md).

## Reproducibility and limitations

The exact dataset source, licenses, dependency versions, performance, artifacts, and commands will be documented and verified in subsequent milestones. **Do not treat this repository as a trained or deployed model yet.** No secrets, checkpoints, model weights, or personal support transcripts belong in Git.

## Usage

Setup, preparation, training, evaluation, inference, demo, and deployment commands will be added as their implementations are tested. Python 3.11 is the target environment; CUDA is optional for training, while CPU inference is the target for the hosted demo.
