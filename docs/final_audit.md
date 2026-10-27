# Final audit — 2026-10-28

## Verified release

| Check | Evidence |
| --- | --- |
| Public model | [`saman-shahroudi/tinydevassist`](https://huggingface.co/saman-shahroudi/tinydevassist), anonymous Hub metadata at `7a2c7cf37d1028c64fc1e74355bffaaa19486129` |
| Public demo | [Gradio ZeroGPU Space](https://huggingface.co/spaces/saman-shahroudi/tinydevassist-demo), revision `d09419f3efb10526fd4bbe93a93ddeb83acfb83c`, `RUNNING` on `zero-a10g` |
| Hosted inference | Anonymous `smoke.py` request returned HTTP 200 and a nonempty answer on 2026-10-28 |
| Previous GitHub CI | [Deployment run 40598949071](https://github.com/samanshahroudi/tinydevassist/actions/runs/40598949071) completed successfully |
| Local checks | Ruff lint and format, 53 pytest cases, `pip check`, and a dry-run training plan passed; 49 relative Markdown links resolved |

The prepared seed has source SHA-256 `92ffc379ffe09d71052d9a7cfc9c9b4fe740c540cb7955e4347e2f646bef151f` and fixed 24/6/6 split counts. The reported six-step CPU LoRA run and paired six-prompt comparison use upstream revision `a10cc1512eabd3dde888204e902eca88bddb4951`; see [evaluation](evaluation.md) for measured results and [text inspection](qualitative_review.md) for errors. The live smoke verifies execution only. It does not measure answer quality, uptime, or quota headroom.

## Remaining limits and deliberate choices

- The 36 project-authored examples and six same-source held-out questions do not support a generalization or quality-improvement claim. An independently authored, expert-rated set is the next scientific step.
- Training uses Transformers `Trainer` with PEFT rather than TRL `SFTTrainer`; [ADR-012](decisions.md) explains the label-masking and version-stability choice. TRL is not required by the executed path.
- The alternate CPU Docker image has not been built or deployed. The verified public path is Gradio ZeroGPU, which has request quotas and may queue users.
- CI runs offline unit checks, not model downloads or GPU inference. Public inference was checked separately; the reproducibility commands refuse to overwrite existing reports and should be run in a fresh checkout for a new experiment.
