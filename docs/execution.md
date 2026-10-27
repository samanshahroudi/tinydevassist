# 3-day execution milestones

Status: **M3 completed in restored isolated workspace**. Progress is checkpointed in Git. Each future session must read Git status, log, active branch, and docs before editing; work on the next unfinished milestone only; test, scan staged diff for secrets/weights, then commit meaningful changes with the configured author. Do not reset or overwrite existing changes.

1. [x] Inspect workspace, create project scope/architecture/decision records.
2. [x] Python project, tooling, CI-ready tests, local environment.
3. [x] Document dataset and licensing; inspect base model metadata (runtime inspection pending).
4. [ ] Deterministic data preparation/splitting and tests.
5. [ ] Baseline inference and resource metrics.
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
