# Architecture decision records

## ADR-001 — A narrow developer-support assistant

- **Decision:** Build TinyDevAssist for concise Git/Python/API/backend-support answers.
- **Context:** A 3-day, laptop-sized project must still demonstrate the complete ML lifecycle.
- **Alternatives considered:** General coding chatbot, customer-service assistant, documentation retrieval service.
- **Reason:** Task breadth can be controlled and measured on small labeled examples; topic aligns with backend engineering.
- **Consequences:** Limited domain, modest expected answer quality, explicit failure analysis required.

## ADR-002 — SmolLM2-360M-Instruct and LoRA as the initial candidate

- **Decision:** Start with `HuggingFaceTB/SmolLM2-360M-Instruct` and LoRA adapters, subject to licensing and hardware checks.
- **Context:** The project must run on a personal computer and serve via CPU Space.
- **Alternatives considered:** TinyLlama 1.1B, larger 1B+ models, full fine-tuning.
- **Reason:** Smaller inference footprint and efficient parameter adaptation.
- **Consequences:** Lower capacity than larger models; determine actual runtime and response quality empirically.

## ADR-003 — Docker-hosted Gradio interface

- **Decision:** Target a Docker Space on port 7860 after local verification.
- **Context:** Recruiters need a reproducible demo with pinned runtime dependencies.
- **Alternatives considered:** Standard Gradio Space, external GPU endpoint.
- **Reason:** A Dockerfile exposes environment and runtime design without requiring paid GPU serving.
- **Consequences:** Must validate CPU latency/memory and provide a health/smoke check.

## ADR-004 — Treat experimental artifacts as unverified until measured

- **Decision:** Label training and deployment status explicitly, never fabricate measurements or URLs.
- **Context:** Training depends on hardware and deployment on external authentication.
- **Alternatives considered:** Publishing demo placeholders as results.
- **Reason:** CV credibility and reproducibility.
- **Consequences:** Missing experiments and blockers remain visible in README and reports.

## ADR-005 — Lightweight package core with optional ML extras

- **Decision:** Package the project using a `src/` layout, setuptools, Python 3.11 as the ML target, and separate `dev`, `train`, and `app` dependency groups.
- **Context:** The baseline CI and developer tooling should run quickly without downloading PyTorch and model artifacts; GPU and CPU deployment have different requirements.
- **Alternatives considered:** A single requirements file with every ML library, Poetry-only packaging, a pinned CUDA wheel in the repository.
- **Reason:** Optional extras make the code importable and testable on ordinary machines while allowing independent runtime environments.
- **Consequences:** Training and serving dependencies are not yet resolved to a tested lockfile; dependency compatibility must be confirmed during implementation and experimentation.

## ADR-006 — Offline environment doctor and lightweight CI

- **Decision:** Add an offline `tinydevassist doctor` diagnostic, Ruff/pytest checks, and a GitHub Actions quality job on Python 3.11.
- **Context:** Each milestone needs quick verifiable quality gates before costly model downloads or training.
- **Alternatives considered:** Downloading the model as a CI smoke test, running a full training step in CI, skipping checks until inference exists.
- **Reason:** Fast deterministic checks are affordable and do not require secrets or specialized hardware.
- **Consequences:** Passing the current CI does not prove ML dependency compatibility, training, inference, or deployed Space health; these require separate later checks.
