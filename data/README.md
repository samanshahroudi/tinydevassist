# Data registry (planned)

Use original or explicitly licensed technical-support question/answer pairs. Before adding data, record source URL, license, revision, size, transformation, and whether redistribution is permitted. Never commit private support tickets, personal data, or API secrets. Raw and processed generated datasets are ignored by default; small, fully attributed sample fixtures may live under `tests/fixtures/`.

Planned preparation: normalize records, enforce schema and content limits, deduplicate, stratify where feasible, split deterministically at seed 42, persist hashes and counts, check leakage across splits, and keep the test split completely out of training.
