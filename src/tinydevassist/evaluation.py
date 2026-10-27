"""Held-out base-versus-adapter generation comparison utilities."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from tinydevassist.data import prompt_messages


@dataclass(frozen=True)
class EvaluationGeneration:
    response: str
    latency_seconds: float
    input_tokens: int
    generated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_generations(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    """Summarize measured generations without inventing a quality score."""
    values = [row[key] for row in rows]
    latencies = [item["latency_seconds"] for item in values]
    return {
        "samples": len(values),
        "nonempty_responses": sum(bool(item["response"].strip()) for item in values),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "mean_generated_tokens": sum(item["generated_tokens"] for item in values) / len(values),
    }


def comparison_report(
    rows: list[dict[str, Any]], *, model_id: str, revision: str, test_sha256: str
) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation needs at least one held-out example")
    for row in rows:
        if not isinstance(row.get("id"), str) or set(row) != {"id", "base", "adapter"}:
            raise ValueError("evaluation rows must contain id, base, and adapter")
        for key in ("base", "adapter"):
            item = row[key]
            if not isinstance(item, dict) or not isinstance(item.get("response"), str):
                raise ValueError("evaluation generation is malformed")
            if not all(
                isinstance(item.get(metric), (int, float)) and math.isfinite(item[metric])
                for metric in ("latency_seconds", "input_tokens", "generated_tokens")
            ):
                raise ValueError("evaluation metrics must be finite")
    return {
        "schema_version": 1,
        "status": "measured_comparison",
        "model_id": model_id,
        "revision": revision,
        "input_sha256": test_sha256,
        "methodology": {
            "prompts": "all held-out test instructions",
            "decoding": "greedy, max_new_tokens=96, temperature=0",
            "quality_metric": None,
            "quality_note": "Responses require human review; no automatic quality claim is made.",
        },
        "summary": {
            "base": summarize_generations(rows, "base"),
            "adapter": summarize_generations(rows, "adapter"),
        },
        "results": rows,
    }


class AdapterGenerator:
    """Load a PEFT adapter and generate one response at a time."""

    def __init__(
        self, model_id: str, revision: str, adapter_path: str, device: str = "cpu"
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the train dependencies for adapter evaluation") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()

    def generate(self, prompt: str, *, max_new_tokens: int = 96) -> EvaluationGeneration:
        encoded = self.tokenizer.apply_chat_template(
            prompt_messages(prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = output[0][encoded["input_ids"].shape[-1] :]
        return EvaluationGeneration(
            response=self.tokenizer.decode(completion, skip_special_tokens=True).strip(),
            latency_seconds=round(time.perf_counter() - started, 6),
            input_tokens=int(encoded["input_ids"].shape[-1]),
            generated_tokens=int(len(completion)),
        )
