# Text inspection of the six held-out generations

This review reads the ignored `outputs/evaluation.json` from the pinned CPU
experiment and compares it with the six reference answers in
`data/processed/test.jsonl`. It is an inspection of this small authored set,
not an independent human rating or a generalization score. The generation
configuration was greedy decoding with at most 96 new tokens. Some answers
stop mid-sentence at that cap.

| ID | Observed base response | Observed adapter response | Technical assessment |
| --- | --- | --- | --- |
| `api-001` | Recommends a retry policy without addressing duplicate orders. | Again recommends retries and omits idempotency keys or atomic uniqueness. | Both miss the central safety mechanism. |
| `debugging-004` | Treats `gc.get_count() > 0` as evidence of a leak. | Invents `psutil.memory_profiler`, which is not a `psutil` API. | Both give unreliable debugging advice; the adapter includes a fabricated call. |
| `git-005` | Suggests `git log --graph` instead of identifying the line's commit. | Suggests `git log --oneline --all --pretty=format:%an` and falsely says it gives line numbers. | Both miss `git blame`; the adapter command is misleading. |
| `http-005` | Correctly associates 201 with creation, but implies 200 cannot be used. | Repeats essentially the same claim. | Partly correct, but misses the `Location` header and the distinction between permitted success responses. |
| `ops-003` | Suggests checking status and logs but spends output on generic network checks. | Suggests restarting the pod before inspecting the previous container logs. | The adapter gives a premature operational action and misses `kubectl logs --previous`. |
| `python-003` | Correctly defines a missing dictionary key. | Correctly defines the missing key with an analogy. | A representative understandable response; both omit a concrete recovery step such as `.get()`. |

The most useful displayed success is the basic `KeyError` definition. The
memory-leak and Git answers are representative failures. In particular,
`psutil.memory_profiler` should **not** be copied into a user workflow and the
suggested `git log --pretty=format:%an` command does not identify a line's
origin. The adapter should be shown as a limited experiment until a larger,
independently written prompt set receives expert review.

## Verbatim adapter excerpts

- `python-003`, prompt: “What does KeyError mean when reading a dictionary?”
  Answer: “A KeyError occurs when you try to access a key in a dictionary that
  does not exist. It's like trying to find a book with a number that doesn't
  exist in your library.” This definition is understandable, though it gives
  no practical recovery step.
- `debugging-004`, prompt: “How can I identify a memory leak in a service?”
  Answer excerpt: “Use the `psutil.memory_profiler` function to get the memory
  usage of the service.” This function does not exist in `psutil`.
- `git-005`, prompt: “How can I find which commit introduced a line?”
  Answer excerpt: “Run the following command in your terminal:
  `git log --oneline --all --pretty=format:%an` ... This command will display
  the commit hash and the line number where the line was introduced.” That
  command formats author names and does not identify the line's commit.

The two failure excerpts were cut at sentence boundaries for display; the
complete responses remain in the ignored experiment report. The model's
96-token cap truncated some complete responses mid-sentence.

The Gradio demo offers four curated prompts spanning Git, Python imports, HTTP
timeouts, and missing API resources. These are examples of the interface,
not endorsements of its answers. The demo now rejects a question if its
tokenized prompt plus requested new tokens exceeds the 512-token sequence
budget used during fine-tuning.
