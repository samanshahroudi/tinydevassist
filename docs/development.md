# Development setup and verification

## Supported environments

- Target for training and serving: **Python 3.11** (3.12 may work; not yet end-to-end tested).
- Python 3.13 can run the dependency-free core and its tests, but ML-library compatibility is **not yet verified**.
- Linux/macOS/WSL2 recommended. GPU is optional for the initial project setup, but training time depends heavily on CUDA availability.
- Do not install unverified CUDA-specific PyTorch wheels blindly. Follow the [official PyTorch selector](https://pytorch.org/get-started/locally/) for the local CUDA driver/runtime.

## Bootstrap (Ubuntu / WSL2)

```bash
cd tiny-llm-hf
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m tinydevassist doctor
make check
```

The `requirements.txt` file provides a minimal editable installation. Optional dependency groups are maintained in `pyproject.toml`:

```bash
python -m pip install -e '.[train]'
python -m pip install -e '.[app]'
python -m pip install -e '.[notebooks]'
```

Do **not** install all extras on a CPU-only deployment until actual serving requirements have been measured. No Hugging Face token is required for environment diagnostics or unit tests.

## Quality gates

```bash
make lint
make format-check
make test
make check
```

`make format` intentionally changes formatting; inspect its diff before committing. The `doctor` command checks module presence only, not dependency compatibility or successful model loading.

## Reproducibility boundary

Version ranges are currently specified, but no resolved dependency lockfile or training environment has been validated. Record a lockfile or `pip freeze`, PyTorch/CUDA versions, dataset hash, seed, model revision, and hardware details when the first real experiment runs. Never present the current scaffold as proof of training or deployment.

## Baseline experiment

```bash
python prepare_data.py --seed 42
python -m pip install -e '.[train,notebooks]'
python inference.py --device cpu --max-samples 6 --temperature 0 --max-new-tokens 96
# Writes ignored outputs/baseline.json on success only.
```

The model and tokenizer are loaded with `trust_remote_code=False`. The script
uses the upstream tokenizer chat template, generates from **held-out test rows**,
and records per-question output, token counts, wall-clock generation latency,
model parameter count/bytes, optional process RSS snapshots (`psutil`), and CUDA
peak allocation if applicable. CPU RSS before/after are not CPU peak memory.
`--revision` should be an immutable upstream commit hash; the report records
both requested and resolved revisions. `main` is convenient but not pinned.
This environment could not download Transformers or model weights, so **no real
baseline metrics or generations exist yet**. The CLI fails rather than
producing placeholder numbers when dependencies are missing.
