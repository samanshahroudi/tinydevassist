# 3-day execution milestones

Status: **M2 completed in isolated workspace**. Progress is checkpointed in Git. Each future session must read Git status, log, active branch, and docs before editing; work on the next unfinished milestone only; test, scan staged diff for secrets/weights, then commit meaningful changes with the configured author. Do not reset or overwrite existing changes.

1. [x] Inspect workspace, create project scope/architecture/decision records.
2. [x] Python project, tooling, CI-ready tests, local environment.
3. [ ] Document dataset and licensing; inspect base model.
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
