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

## ADR-007 — Original, transparent seed dataset instead of externally scraped Q&A

- **Decision:** Track 36 newly authored illustrative developer-support pairs in `data/samples/developer_support_v1.jsonl`, under Apache-2.0, with a dataset card and stable IDs.
- **Context:** The project needs a small legally redistributable dataset and an auditable starting point within three days. External Q&A corpora can contain third-party rights, personal information, and mixed licensing.
- **Alternatives considered:** Stack Overflow exports, GitHub issue threads, public technical-support corpora, or untracked synthetic data.
- **Reason:** Explicitly authored examples avoid unverifiable scraping provenance, are small enough to inspect, and allow the pipeline to be tested without network access.
- **Consequences:** Drafted with language-model assistance, the examples require independent technical review; 36 rows are insufficient for meaningful generalization claims. An independent held-out evaluation set is still needed. No train/validation/test split exists yet.

## ADR-008 — Verify upstream model metadata before downloading weights

- **Decision:** Retain `HuggingFaceTB/SmolLM2-360M-Instruct` as the candidate and record its official Apache-2.0 license, English focus, chat-template usage, and limitations in `docs/model_selection.md`.
- **Context:** Model metadata can be checked independently of local GPU and package availability.
- **Alternatives considered:** Switching immediately to TinyLlama 1.1B or SmolLM2-135M-Instruct; downloading the model before documenting license and scope.
- **Reason:** The 360M class balances capacity with a smaller anticipated serving footprint, while retaining an explicit upstream license.
- **Consequences:** No inference, training, memory, or latency claims are made. The floating `main` revision must be resolved to a commit SHA before a reproducible experiment.

## ADR-009 — Deterministic, stratified small-data splits and chat messages

- **Decision:** Normalize source rows (NFC/whitespace), reject ambiguous duplicate answers and near-duplicate questions, deduplicate identical questions/answers, and split by category with seed 42. Store generated JSONL and a SHA-256 manifest outside Git. Encode SFT examples as role/content messages, not hard-coded model tokens.
- **Context:** The 36-row project-authored seed needs reproducible preprocessing, leakage checks, and a clean contract for later Transformers/TRL training.
- **Alternatives considered:** Unseeded random splits, a single global random split, committing derived datasets, manually concatenating chat tokens, and silently removing similar but potentially distinct questions.
- **Reason:** Per-category stratification retains coverage in all three splits; hashing and stable ordering support auditability; rejecting near-duplicates avoids accidental leakage and data loss; tokenizer chat templates preserve model-specific formatting.
- **Consequences:** The current seed produces 24/6/6 rows (4/1/1 per category). Near-duplicate detection is heuristic and may need manual review for larger datasets. Generated files must be regenerated after source changes. Held-out rows from one authored source do not constitute an independent generalization benchmark.
