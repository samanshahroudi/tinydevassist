"""Offline baseline tests; do not download model weights in CI."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from inference import load_examples, run_baseline

from tinydevassist.inference import BaselineGenerator


class FakeTokens:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self


class FakeParameter:
    def data_ptr(self):
        return 123

    def numel(self):
        return 4

    def element_size(self):
        return 4


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    init_kwargs = {}

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        return FakeTokens([1, 2, 3])

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return "use git status" if tokens == [4, 5] else "unexpected"


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(_commit_hash="abc123")
        self.last_kwargs = None

    def parameters(self):
        return iter([FakeParameter(), FakeParameter()])

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        assert input_ids.shape == (1, 3)
        return [[1, 2, 3, 4, 5]]


class FakeTorch:
    __version__ = "fake-test"
    cuda = SimpleNamespace(is_available=lambda: False)

    @staticmethod
    def inference_mode():
        return nullcontext()


def make_generator():
    return BaselineGenerator(
        "test/model", tokenizer=FakeTokenizer(), model=FakeModel(), torch_module=FakeTorch()
    )


def test_chat_template_greedy_generation_and_metrics():
    generator = make_generator()
    result = generator.generate("How do I check Git status?")
    assert result.response == "use git status"
    assert result.metrics.input_tokens == 3
    assert result.metrics.generated_tokens == 2
    assert result.metrics.model_parameter_count == 4
    assert result.metrics.model_parameter_bytes == 16
    assert result.metrics.cuda_peak_allocated_bytes is None
    assert generator.model.last_kwargs["do_sample"] is False
    assert "temperature" not in generator.model.last_kwargs
    assert generator.resolved_revision == "abc123"


def test_sampling_controls():
    generator = make_generator()
    generator.generate("Git?", max_new_tokens=12, temperature=0.6)
    assert generator.model.last_kwargs["temperature"] == 0.6
    assert generator.model.last_kwargs["max_new_tokens"] == 12


@pytest.mark.parametrize("tokens,temperature", [(0, 0), (513, 0), (5, -0.1), (5, 2.1)])
def test_invalid_generation_controls(tokens, temperature):
    with pytest.raises(ValueError):
        make_generator().generate("Git?", max_new_tokens=tokens, temperature=temperature)


def test_rejects_blank_prompt_and_unsupported_device():
    with pytest.raises(ValueError):
        make_generator().generate(" ")
    with pytest.raises(ValueError):
        BaselineGenerator("test/model", device="mps", torch_module=FakeTorch())


def test_baseline_report_has_real_mocked_measurements():
    report = run_baseline(
        make_generator(),
        [{"id": "git-001", "instruction": "Git?"}],
        max_new_tokens=12,
        temperature=0,
    )
    assert report["status"] == "measured"
    assert report["summary"]["samples"] == 1
    assert report["results"][0]["id"] == "git-001"
    assert report["results"][0]["response"] == "use git status"


def test_load_examples(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"a","instruction":"first"}\n{"id":"b","instruction":"second"}\n',
        encoding="utf-8",
    )
    assert load_examples(path, 1) == [{"id": "a", "instruction": "first"}]
    with pytest.raises(ValueError):
        load_examples(path, 0)
