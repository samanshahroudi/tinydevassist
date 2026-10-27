# 3-day execution milestones

Status: **M4 completed; M5 baseline implementation committed, real measurements blocked**. Progress is checkpointed in Git. Each future session must read Git status, log, active branch, and docs before editing; work on the next unfinished milestone only; test, scan staged diff for secrets/weights, then commit meaningful changes with the configured author. Do not reset or overwrite existing changes.

1. [x] Inspect workspace, create project scope/architecture/decision records.
2. [x] Python project, tooling, CI-ready tests, local environment.
3. [x] Document dataset and licensing; inspect base model metadata (runtime inspection pending).
4. [x] Deterministic data preparation/splitting and tests.
5. [ ] Baseline inference and resource metrics (code and notebooks implemented; real model run pending).
6. [ ] Configurable LoRA fine-tuning pipeline.
7. [ ] Execute truthful, bounded training experiment.
8. [ ] Base-versus-adapted evaluation.
9. [ ] Refine inference controls and examples.
10. [ ] Gradio app and smoke tests.
11. [ ] Docker, Space metadata, deployment guide.
12. [ ] Publish model and Space, verify only if credentials available.
13. [ ] Model card, experiment report, architecture finalization.
14. [ ] CI, cleanup, reproduction instructions, CV bullet.
15. [ ] Full final verification and audit.

## Session 1 report

- Milestone: 1
- Workspace: isolated project in `/mnt/data/tiny-llm-hf`; no existing Git repository found in accessible task workspace.
- Created: README, architecture, decision records, milestone tracker, Git ignore rules.
- Verification: Git initialization, status/diff, author identity, committed-file listing.
- Limitations: no user machine access established, no GitHub remote, no Hugging Face authentication, no model download or GPU capability test.

## Session 2 report

- Milestone: 2 — Python packaging, optional dependency groups, development commands, offline diagnostics, tests, and lightweight CI.
- Workspace: `/mnt/data/tiny-llm-hf` in an isolated execution container; **not the user's laptop**. The Milestone 1 Git history was found intact on `main`, with a clean initial working tree.
- Added: `pyproject.toml`, `requirements.txt`, `Makefile`, `src/tinydevassist/`, `tests/`, `.github/workflows/ci.yml`, `.env.example`, `docs/development.md`, `configs/README.md`, `notebooks/README.md`.
- Verification: editable core install succeeded; `python -m pytest -q` passed (3 tests); `python -m compileall -q src tests` passed; `python -m tinydevassist doctor` produced valid JSON; `git diff --check` passed. Ruff lint/format checks could not run: Ruff was not installed and its wheel was unavailable from the configured package source. Python 3.13 is available here, whereas Python 3.11 is the recommended ML environment.
- Remaining: no model download, training, deployment, remote GitHub repository, or access to the user's computer established.

## Session 3 report

- Milestone: 3 — documented model selection, verified upstream metadata, and created a tracked, auditable 36-record seed dataset with an explicit dataset card and licensing policy.
- Workspace: restored the previously committed Milestone 2 archive to `/mnt/data/tiny-llm-hf-m3-work/tiny-llm-hf` without overwriting the unrelated existing README in `/mnt/data/tiny-llm-hf`; restored branch `main` and verified clean status and both previous commits before editing. **Not the user's laptop.**
- Changed: `LICENSE`, `README.md`, `configs/model.json`, `data/README.md`, `data/DATASET_CARD.md`, `data/samples/developer_support_v1.jsonl`, `docs/model_selection.md`, `docs/decisions.md`, `docs/execution.md`, and `tests/test_dataset_sample.py`.
- Verification: `PYTHONPATH=src python -m pytest -q` (5 passed); `python -m compileall -q src tests` passed; `git diff --check` passed. Ruff was not installed and cannot be claimed as passing. Staged diff, secret, and weight checks are required before commit.
- Limitations: no upstream model weights downloaded, no model inference or training, no GitHub remote, no Hugging Face credentials, no user-computer access. `main` remains an unpinned upstream revision until an experiment is executed.

## Session 4 report

- Milestone: 4 — implemented deterministic source validation, normalization, duplicate/near-duplicate controls, category-stratified splits, chat-message formatting, leakage checks, and SHA-256 manifest.
- Workspace: restored Milestone 3 archive into `/mnt/data/tinydevassist-m4-work/tiny-llm-hf` to avoid overwriting an unrelated `/mnt/data/tiny-llm-hf/README.md`; no access to the user's personal computer. Verified `main`, clean initial status, and all three previous commits before edits.
- Files: `prepare_data.py`, `src/tinydevassist/data.py`, `tests/test_data_preparation.py`, `Makefile`, `README.md`, `data/README.md`, `data/DATASET_CARD.md`, `docs/architecture.md`, `docs/decisions.md`, and this tracker.
- Data result: source SHA-256 `92ffc379ffe09d71052d9a7cfc9c9b4fe740c540cb7955e4347e2f646bef151f`; 36 retained rows, 24/6/6 stratified split, no duplicate questions or IDs. Generated outputs are Git-ignored.
- Verification: offline tests, Python compilation, manifest/hash checks, staged diff/secret/weights checks, and Git commit are recorded in the run report. Ruff remains unavailable in this environment and must not be reported as passing.
- Limitations: no real model inference/training/deployment, no user-laptop access, no GitHub remote, no Hugging Face credentials. Independent evaluation questions still needed.

## Session 5 report

- Milestone: 5 **implementation checkpoint** — added baseline CLI, lazy-loaded Transformers generator, deterministic decoding and measurable JSON output, offline tests, and the first two notebooks. Real baseline measurements remain **uncompleted**, so Milestone 5 stays unchecked.
- Workspace: restored Milestone 4 archive to `/mnt/data/tinydevassist-work-m5/tiny-llm-hf`, separate from an unrelated `/mnt/data/tiny-llm-hf/README.md`. Verified `main`, clean initial status, and commits `4bedf18`, `4c0b4a0`, `3fbd101`, `d39f6ca`. Not the user's laptop.
- Files: `inference.py`, `src/tinydevassist/inference.py`, `tests/test_inference.py`, `notebooks/01_dataset_exploration.ipynb`, `notebooks/02_baseline_model.ipynb`, `notebooks/README.md`, `README.md`, `docs/development.md`, `docs/decisions.md`, `docs/execution.md`, `pyproject.toml`, `Makefile`, `.gitignore`.
- Verification: 24 offline tests passed; compilation passed; notebook 01 executed against the tracked source and notebook 02 executed in **skip mode**; baseline CLI correctly fails when Transformers is unavailable. No real model weights downloaded or executed. Ruff unavailable and offline installation unsuccessful; do not claim lint or format checks passed.
- Blockers: no network resolution for huggingface.co; Transformers/weights unavailable; no local-computer access or GitHub remote. Actual model performance, baseline generations, training, and deployment remain unverified.
