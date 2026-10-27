# Hugging Face deployment

## What is packaged

The model repository contains the PEFT adapter, tokenizer files, and a model
card. The Docker Space contains the Gradio app and the small package modules it
needs. `deploy_hf.py` uses an explicit allowlist; it excludes raw experiment
reports, training arguments, secrets, and unrelated files from both uploads.
The Space runs as a non-root user on port 7860 and loads the pinned base model
plus the public adapter repository on its first inference request.

## Local setup and smoke test

```powershell
python -m pip install -e '.[app,dev]'
$env:TINYDEVASSIST_ADAPTER_PATH='checkpoints/tinydevassist-lora'
python app.py
# In a second shell:
python smoke.py --url http://127.0.0.1:7860
```

The local Docker daemon is required for a Docker build test:

```powershell
docker build -f space/Dockerfile -t tinydevassist-space .
docker run --rm -p 7860:7860 -e HF_MODEL_REPO=<published-model-id> tinydevassist-space
```

## Publish

Choose explicit Hugging Face repository IDs, for example
`your-name/tinydevassist` (model) and `your-name/tinydevassist-demo` (Space).
Authenticate locally; never paste a token into a commit or chat.

```powershell
.venv\Scripts\hf.exe auth login
python deploy_hf.py --model-repo <namespace/model-name> --space-repo <namespace/space-name>
python deploy_hf.py --model-repo <namespace/model-name> --space-repo <namespace/space-name> --execute
```

The dry run checks the exact curated file manifest. Execution requires a valid
Hub login, creates new public repositories, uploads the adapter and Space files,
and sets `HF_MODEL_REPO` as a Space variable. The command refuses to overwrite
existing repositories; if a partial upload created one, inspect it before
recovery. Do not interpret an upload message as a healthy deployment.

After publishing, check the Space build and runtime logs, then run:

```powershell
python smoke.py --url https://<namespace>-<space-name>.hf.space
```

Record verified public URLs and smoke results in the README only after this
check succeeds. The adapter uses roughly 1.45 GB of base fp32 parameter memory
plus process overhead; Space CPU memory and cold start must be measured in the
actual deployment. The six held-out prompts are too small to establish quality.

## Current publication state and recovery

The public adapter repository is
[`saman-shahroudi/tinydevassist`](https://huggingface.co/saman-shahroudi/tinydevassist). An
anonymous Hub query verified its model card, adapter configuration, adapter
weights, and tokenizer files at Hub commit
`7e21c7d08913bc0725ea3cf2c6b72b184f694572`. The Docker Space
`saman-shahroudi/tinydevassist-demo` does not exist. The prior attempt under
the former `lnxsam` namespace returned HTTP 402:
Hugging Face requires a PRO subscription for Gradio/Docker Spaces on free
`cpu-basic` for this account. No Space build or public inference ran.

After the account has the required hosting entitlement, resume with:

```powershell
.venv\Scripts\python.exe deploy_hf.py --model-repo saman-shahroudi/tinydevassist --space-repo saman-shahroudi/tinydevassist-demo --space-only
.venv\Scripts\python.exe deploy_hf.py --model-repo saman-shahroudi/tinydevassist --space-repo saman-shahroudi/tinydevassist-demo --space-only --execute
```

The resume mode verifies every curated published model file against the local
artifact by SHA-256 before creating the Space; it never replaces the existing
model. Then inspect the Space build and runtime logs and run
`python smoke.py --url https://saman-shahroudi-tinydevassist-demo.hf.space` only after
the actual Space URL is confirmed. A failed upload or build still does not
count as a deployment.

The application and publisher read `HF_MODEL_REPO` and `HF_SPACE_REPO` rather
than hard-coding a user namespace. These variables are unset in this workspace.
If they were set in another shell or in Space settings, change them to the
new canonical IDs before running a deployment; the local app can alternatively
use `TINYDEVASSIST_ADAPTER_PATH` for its ignored adapter directory.
