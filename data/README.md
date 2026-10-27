# TinyDevAssist dataset registry

The first dataset candidate is [`samples/developer_support_v1.jsonl`](samples/developer_support_v1.jsonl): **36 original illustrative question/answer pairs** written for this project (six each in `git`, `python`, `http`, `api`, `debugging`, `ops`). The records were drafted with language-model assistance, are **not** real support transcripts, and have not yet undergone an independent technical review. This is a reproducible **seed dataset**, not evidence of production quality or a sufficient sample for strong generalization.

Read the [dataset card](DATASET_CARD.md) before using or redistributing it. The dataset is published under the repository's Apache-2.0 license, separately from the upstream base model. Do not mix in third-party documentation, code, private tickets, or personal data without provenance and licensing review.

## Raw record format

One UTF-8 JSON object per line:

```json
{"id":"git-001","category":"git","instruction":"...","response":"...","source":"project-authored-v1"}
```

Required fields are `id`, `category`, `instruction`, `response`, and `source`. IDs must be stable and unique. Prompts and responses must be nonempty English strings. The file is tracked so reviewers can audit exact source examples; derived train/validation/test files belong in ignored `data/processed/` and will be generated in Milestone 4.

## Planned preparation (not implemented yet)

Normalize whitespace, validate schema and length, deduplicate questions and near-duplicates, then create deterministic seed-42 splits with a held-out test set. Keep split counts, source hashes, and leakage checks in a manifest. Do not train on held-out examples. Independent test questions and human review should be added before making meaningful quality claims.
