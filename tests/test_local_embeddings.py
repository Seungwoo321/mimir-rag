import asyncio
import math
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mimir_rag.config import Settings
from mimir_rag.errors import ProviderError
from mimir_rag.local_embeddings import LocalEmbeddingEngine


class Tokenizer:
    def __init__(self) -> None:
        self.prefixes: list[str] = []
        self.coverage: list[int] = []

    def encode(self, text: str, **options: Any) -> list[int]:
        if text in {"query: ", "passage: "}:
            self.prefixes.append(text)
            return [900]
        assert options.get("truncation") is False
        return [int(number) for number in text.split()]

    def num_special_tokens_to_add(self, **options: Any) -> int:
        return 2

    def build_inputs_with_special_tokens(self, tokens: list[int]) -> list[int]:
        return [999] + tokens + [999]

    def prepare_for_model(self, tokens: list[int], **options: Any) -> dict[str, list[int]]:
        assert options["truncation"] is False
        assert options["return_special_tokens_mask"] is False
        self.coverage.extend(tokens[1:])
        return {
            "input_ids": [999] + tokens + [999],
            "attention_mask": [1] * (len(tokens) + 2),
        }

    def pad(self, features: list[dict[str, list[int]]], **options: Any) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            key: torch.tensor(
                [feature[key] + [0] * (width - len(feature[key])) for feature in features]
            )
            for key in features[0]
        }


class Model:
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.batches: list[tuple[int, int]] = []

    def __call__(self, **inputs: torch.Tensor) -> Any:
        ids = inputs["input_ids"].float()
        self.batches.append(tuple(ids.shape))
        hidden = torch.stack([ids, torch.ones_like(ids)], dim=-1)
        if self.mode == "zero":
            hidden.zero_()
        if self.mode == "nan":
            hidden.fill_(float("nan"))
        return SimpleNamespace(last_hidden_state=hidden)


def engine(*, mode: str = "normal", dimensions: int = 2) -> LocalEmbeddingEngine:
    result = LocalEmbeddingEngine(
        Settings(
            embedding_dimensions=dimensions,
            embedding_window_tokens=32,
            embedding_inference_batch_size=2,
        )
    )
    result._tokenizer = Tokenizer()
    result._model = Model(mode=mode)
    result._torch = torch
    return result


async def test_all_tokens_including_tail_are_covered_once_with_bounded_batches() -> None:
    instance = engine()
    long = list(range(1, 101))
    actual = await instance.embed([" ".join(map(str, long)), "200 201"], purpose="query")
    assert instance._tokenizer.coverage == long + [200, 201]
    assert instance._tokenizer.prefixes == ["query: "]
    assert all(rows <= 2 and width <= 32 for rows, width in instance._model.batches)
    windows = [long[start : start + 29] for start in range(0, len(long), 29)]
    weighted = sum(
        (sum(window) + 999 + 900 + 999) / (len(window) + 3) * len(window) for window in windows
    ) / len(long)
    short_mean = (200 + 201 + 999 + 900 + 999) / 5
    for vector, mean in zip(actual, [weighted, short_mean], strict=True):
        norm = math.sqrt(mean * mean + 1)
        assert vector == pytest.approx([mean / norm, 1 / norm])
    await instance.embed(["7"], purpose="document")
    assert instance._tokenizer.prefixes[-1] == "passage: "


@pytest.mark.parametrize("mode,dimensions", [("zero", 2), ("nan", 2), ("normal", 3)])
async def test_invalid_output_is_rejected(mode: str, dimensions: int) -> None:
    with pytest.raises(ProviderError):
        await engine(mode=mode, dimensions=dimensions).embed(["1 2"])


async def test_empty_and_oversized_inputs_fail_before_loading() -> None:
    instance = LocalEmbeddingEngine(Settings(max_extracted_chars=5))
    assert await instance.embed([]) == []
    for text in [" ", "abcdef"]:
        with pytest.raises(ProviderError):
            await instance.embed([text])
    assert instance._model is None


async def test_model_failure_does_not_leak_input_or_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = LocalEmbeddingEngine(Settings())

    def broken(texts: list[str], purpose: str) -> list[list[float]]:
        raise RuntimeError("private-text-sensitive-detail")

    monkeypatch.setattr(instance, "_encode", broken)
    with pytest.raises(ProviderError) as error:
        await instance.embed(["private-text"])
    assert "private-text" not in str(error.value)


async def test_cancellation_holds_device_lock_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = LocalEmbeddingEngine(Settings())
    entered = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0
    calls = 0

    def blocked(texts: list[str], purpose: str) -> list[list[float]]:
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        entered.set()
        release.wait(timeout=5)
        active -= 1
        return [[1.0]]

    monkeypatch.setattr(instance, "_encode", blocked)
    first = asyncio.create_task(instance.embed(["first"]))
    await asyncio.to_thread(entered.wait, 2)
    first.cancel()
    second = asyncio.create_task(instance.embed(["second"]))
    await asyncio.sleep(0.03)
    assert calls == 1
    assert not first.done()
    first.cancel()
    await asyncio.sleep(0.03)
    first.cancel()
    await asyncio.sleep(0.03)
    assert calls == 1
    assert not first.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == [[1.0]]
    assert peak == 1
    assert active == 0


def test_loader_pins_weights_disallows_remote_code_and_caches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    import transformers

    observed: list[dict[str, Any]] = []
    model = SimpleNamespace(eval=lambda: None, to=lambda device: None)

    def load(name: str, **options: Any) -> Any:
        assert name == "/synthetic-cache/pinned"
        observed.append(options)
        return model if options.get("use_safetensors") else Tokenizer()

    def cached(**options: Any) -> str:
        assert options["local_files_only"] is True
        assert options["repo_id"] == "intfloat/multilingual-e5-small"
        assert options["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        return "/synthetic-cache/pinned"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load)
    instance = LocalEmbeddingEngine(
        Settings(embedding_device="cpu", embedding_local_files_only=True)
    )
    instance._load()
    instance._load()
    assert len(observed) == 2
    assert all(
        options["revision"] == instance.settings.embedding_revision
        and options["trust_remote_code"] is False
        and options["local_files_only"] is True
        for options in observed
    )
    assert observed[1]["use_safetensors"] is True


def test_unavailable_explicit_device_fails_before_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ProviderError, match="unavailable"):
        LocalEmbeddingEngine(Settings(embedding_device="mps"))._load()


async def test_real_fast_tokenizer_integer_windows_do_not_request_unsupported_masks() -> None:
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import XLMRobertaTokenizerFast

    backend = BackendTokenizer(
        WordLevel(
            {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "query": 4,
                ":": 5,
                "passage": 6,
                "hello": 7,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    tokenizer = XLMRobertaTokenizerFast(tokenizer_object=backend)
    assert tokenizer.num_special_tokens_to_add(pair=False) == 2
    with pytest.raises(AssertionError):
        tokenizer.prepare_for_model([7], return_special_tokens_mask=True)
    instance = engine()
    instance._tokenizer = tokenizer
    vectors = await instance.embed(["hello " * 200, "hello"], purpose="query")
    assert len(vectors) == 2
    assert all(
        math.sqrt(sum(number * number for number in vector)) == pytest.approx(1)
        for vector in vectors
    )
    assert len(instance._model.batches) >= 4
    assert all(rows <= 2 and width <= 32 for rows, width in instance._model.batches)
