"""Lazy-loaded Transformers baseline inference with transparent runtime measurements."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from tinydevassist.data import prompt_messages


@dataclass(frozen=True)
class GenerationMetrics:
    latency_seconds: float
    input_tokens: int
    generated_tokens: int
    model_parameter_count: int
    model_parameter_bytes: int
    process_rss_before_bytes: int | None
    process_rss_after_bytes: int | None
    cuda_peak_allocated_bytes: int | None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    response: str
    metrics: GenerationMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _model_size(model: Any) -> tuple[int, int]:
    # Count unique underlying tensors (tied embedding and LM-head weights share storage).
    parameters = list(model.parameters())
    unique: dict[int, Any] = {}
    for parameter in parameters:
        key = parameter.data_ptr()
        unique.setdefault(key, parameter)
    count = sum(parameter.numel() for parameter in unique.values())
    size = sum(parameter.numel() * parameter.element_size() for parameter in unique.values())
    return count, size


class BaselineGenerator:
    """Load a model once, then generate responses using its own chat template."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: str = "auto",
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id and revision must be nonempty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError("Install the train extra for PyTorch inference") from exc
        self.torch = torch_module
        if device == "auto":
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must both be supplied or both omitted")
        if tokenizer is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install the train extra for Transformers inference") from exc
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        )
        self.parameter_count, self.parameter_bytes = _model_size(self.model)

    def generate(
        self, prompt: str, *, max_new_tokens: int = 96, temperature: float = 0.0
    ) -> GenerationResult:
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        messages = prompt_messages(prompt)
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        before = _rss_bytes()
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(input_ids=input_ids, **kwargs)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        completion = generated[0][input_ids.shape[-1] :]
        response = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return GenerationResult(
            prompt=prompt,
            response=response,
            metrics=GenerationMetrics(
                latency_seconds=round(elapsed, 6),
                input_tokens=int(input_ids.shape[-1]),
                generated_tokens=int(len(completion)),
                model_parameter_count=self.parameter_count,
                model_parameter_bytes=self.parameter_bytes,
                process_rss_before_bytes=before,
                process_rss_after_bytes=_rss_bytes(),
                cuda_peak_allocated_bytes=(
                    int(self.torch.cuda.max_memory_allocated())
                    if self.device == "cuda"
                    else None
                ),
            ),
        )
