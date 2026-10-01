from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any, Literal, cast

from .config import Settings
from .errors import ProviderError


class LocalEmbeddingEngine:
    """Pinned, local E5 inference covering every content token through bounded windows."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        options = {
            "revision": self.settings.embedding_revision,
            "trust_remote_code": False,
            "local_files_only": self.settings.embedding_local_files_only,
        }
        device = self.settings.embedding_device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise ProviderError("Requested MPS device is unavailable.")
        model_path: str = self.settings.embedding_model
        if self.settings.embedding_local_files_only:
            model_path = snapshot_download(
                repo_id=model_path, revision=self.settings.embedding_revision, local_files_only=True
            )
        tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_path, **options)
        model = AutoModel.from_pretrained(model_path, use_safetensors=True, **options)
        model.eval()
        model.to(device)
        self._tokenizer, self._model, self._torch, self._device = tokenizer, model, torch, device

    async def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        if purpose not in {"document", "query"}:
            raise ProviderError("Unknown embedding purpose.")
        if not texts:
            return []
        if any(not text.strip() or len(text) > self.settings.max_extracted_chars for text in texts):
            raise ProviderError("Embedding inputs must be nonempty and within the text limit.")
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(self._encode, list(texts), purpose))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Hold the lock until native inference stops to prevent overlapping device work.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()
                raise
            except ProviderError:
                raise
            except Exception:
                raise ProviderError(
                    "Local embedding inference failed; check model cache, dependencies and device."
                ) from None

    def _encode(self, texts: list[str], purpose: str) -> list[list[float]]:
        self._load()
        tokenizer, model, torch = self._tokenizer, self._model, self._torch
        prefix = tokenizer.encode(
            "query: " if purpose == "query" else "passage: ", add_special_tokens=False
        )
        capacity = (
            self.settings.embedding_window_tokens
            - tokenizer.num_special_tokens_to_add(pair=False)
            - len(prefix)
        )
        if capacity <= 0:
            raise ProviderError("Embedding window leaves no content capacity.")
        sums = torch.zeros((len(texts), self.settings.embedding_dimensions), device=self._device)
        pending: list[tuple[int, list[int]]] = []

        def flush() -> None:
            features = []
            for _owner, window in pending:
                feature = tokenizer.prepare_for_model(
                    prefix + window,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_special_tokens_mask=False,
                )
                expected_ids = tokenizer.build_inputs_with_special_tokens(prefix + window)
                expected_length = (
                    len(prefix) + len(window) + tokenizer.num_special_tokens_to_add(pair=False)
                )
                if (
                    feature["input_ids"] != expected_ids
                    or len(expected_ids) != expected_length
                    or feature["attention_mask"] != [1] * expected_length
                ):
                    raise ProviderError("Embedding window token coverage mismatch.")
                features.append(feature)
            padded = tokenizer.pad(features, padding=True, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in padded.items()}
            mask_tensor = inputs["attention_mask"]
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state.float()
                if hidden.shape[-1] != self.settings.embedding_dimensions:
                    raise ProviderError("Local model dimensions differ from configured dimensions.")
                window_means = (hidden * mask_tensor.unsqueeze(-1)).sum(dim=1)
                window_means /= mask_tensor.sum(dim=1).unsqueeze(-1)
                for row, (owner, window) in enumerate(pending):
                    sums[owner] += window_means[row] * len(window)
            pending.clear()

        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(
                text, add_special_tokens=False, truncation=False, verbose=False
            )
            if not tokens:
                raise ProviderError("Embedding input produced no content tokens.")
            for start in range(0, len(tokens), capacity):
                pending.append((owner, tokens[start : start + capacity]))
                if len(pending) == self.settings.embedding_inference_batch_size:
                    flush()
        if pending:
            flush()
        results: list[list[float]] = []
        for vector in sums.cpu().tolist():
            norm = math.sqrt(sum(number * number for number in vector))
            if (
                len(vector) != self.settings.embedding_dimensions
                or not math.isfinite(norm)
                or norm <= 0
                or not all(math.isfinite(number) for number in vector)
            ):
                raise ProviderError("Local embedding output has invalid dimensions or norm.")
            results.append([float(number / norm) for number in vector])
        return results
