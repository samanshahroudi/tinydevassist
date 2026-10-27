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
