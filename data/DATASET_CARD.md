---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
pretty_name: TinyDevAssist Seed Support Q&A v1
size_categories:
  - n<1K
---

# Dataset card — TinyDevAssist Seed Support Q&A v1

## Purpose

A small, auditable starter set for supervised instruction-following experiments on short developer-support answers. Intended for engineering demos, **not** production helpdesk deployment or benchmarking general coding competence.

## Provenance and license

- **Source:** 36 newly authored illustrative Q&A examples created for this repository with language-model drafting assistance. They are not sourced from user messages, incident tickets, Stack Overflow, scraped websites, or proprietary documentation.
- **Maintainer:** Saman Shahroudi (project repository).
- **License:** Apache License 2.0; see [`../LICENSE`](../LICENSE). The license applies to these newly authored examples, not to any third-party material or the base model.
- **Version:** `project-authored-v1`, tracked in Git. SHA-256 of the JSONL file: `92ffc379ffe09d71052d9a7cfc9c9b4fe740c540cb7955e4347e2f646bef151f`. Record the Git commit for each run.
- **Source URL:** this repository, `data/samples/developer_support_v1.jsonl` (no public remote URL has been established).
- **Redistribution:** intended to be redistributable under Apache-2.0, subject to the license and ordinary review of any future contributions.
- **Collection date:** 2026-10-27.

## Schema and composition

Fields: `id` (unique stable identifier), `category` (one of `git`, `python`, `http`, `api`, `debugging`, `ops`), `instruction` (question), `response` (target answer), and `source` (`project-authored-v1`). There are 6 records per category and 36 total. All examples are English. No sensitive records or real-world personal information were intentionally included.

## Quality controls and limitations

Examples are short and may encode oversimplifications. Answers include commands that require human judgment before running, especially Git operations. **No independent subject-matter review, model training, train/validation/test split, or evaluation has occurred.** A 36-row sample is too small to substantiate broad model improvement; any training run must be reported as a small demonstration and evaluated on genuinely held-out, preferably independently authored questions.

## Preparation and leakage policy

Milestone 4 will implement deterministic normalization, duplicate checks, seeded splits, and a machine-readable manifest. Test records must never be included in prompts or training examples. Do not evaluate solely by exact-match wording or on training questions.

## Model and intended usage

Candidate model: [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), separately Apache-2.0 licensed. Use its tokenizer chat template; do not assume a hand-written prompt format is equivalent.
