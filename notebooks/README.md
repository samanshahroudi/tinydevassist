# Reproducible experiments

The CLI and `src/tinydevassist/` are the implementation source of truth; notebooks
explore and visualize **real outputs** without duplicating the model pipeline.

| Notebook | Purpose | Verified in this workspace |
| --- | --- | --- |
| [`01_dataset_exploration.ipynb`](01_dataset_exploration.ipynb) | Reproduce the split; plot category and text-length distributions | **Executed** on the tracked 36-row seed dataset; outputs included |
| [`02_baseline_model.ipynb`](02_baseline_model.ipynb) | Run baseline CLI (explicit opt-in) and inspect response/latency table | **Executed in skip mode only**; no model weights or measured results |
| `03_lora_finetuning.ipynb` | Future: inspect a real training run | Not implemented |
| `04_model_evaluation.ipynb` | Future: baseline versus adapted model | Not implemented |
| `05_inference_optimization.ipynb` | Optional future: measured CPU/GPU comparisons | Not implemented |

From the repository root, use a Python 3.11 virtual environment:

```bash
python -m pip install -e '.[dev,notebooks]'
python -m ipykernel install --user --name tinydevassist --display-name TinyDevAssist
jupyter notebook notebooks/
```

For actual model inference, also install `-e '.[train]'` and ensure the upstream
model is downloadable or already cached. In notebook 02 set `RUN_BASELINE=True`;
by default it **does not download weights**. Pin `--revision` to a model commit SHA
for a reproducible report. Real model outputs and metrics are written to ignored
`outputs/baseline.json`, not fabricated or committed.

Notebook outputs may be committed only if generated from the tracked dataset or
real measured experiments, with their provenance documented. Never commit model
weights, credentials, private prompts, or fake benchmark numbers.
