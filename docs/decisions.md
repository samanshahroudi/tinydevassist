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
