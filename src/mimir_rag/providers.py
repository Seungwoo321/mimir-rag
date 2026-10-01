from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import Sequence
from typing import Any, Literal, Self

import httpx

from .config import Settings
from .errors import ConfigurationError, ProviderError
from .tokenization import get_encoding

OPENAI_ROOT = "https://api.openai.com/v1"
ANTHROPIC_ROOT = "https://api.anthropic.com/v1"
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _wire_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_wire_schema(item) for item in value]
    if isinstance(value, dict):
        unsupported = {
            "default",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "pattern",
            "title",
        }
        mappings = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
        children = {
            "items",
            "prefixItems",
            "additionalProperties",
            "unevaluatedProperties",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "allOf",
            "anyOf",
            "oneOf",
        }
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in unsupported:
                continue
            if key in mappings and isinstance(item, dict):
                result[key] = {name: _wire_schema(child) for name, child in item.items()}
            elif key in children:
                result[key] = _wire_schema(item)
            else:
                result[key] = item
        return result
    return value


class ProviderClient:
    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self._http = httpx.AsyncClient(
            timeout=settings.api_timeout_seconds,
            transport=transport,
            limits=httpx.Limits(max_connections=settings.api_concurrency),
            follow_redirects=False,
            trust_env=False,
        )
        self._semaphore = asyncio.Semaphore(settings.api_concurrency)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def _headers(self, provider: str) -> dict[str, str]:
        key = (
            self.settings.openai_api_key
            if provider == "openai"
            else self.settings.anthropic_api_key
        )
        if key is None or not key.get_secret_value().strip():
            name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
            raise ConfigurationError(
                f"Set {name} in the process environment before cloud inference."
            )
        if any(not 33 <= ord(character) <= 126 for character in key.get_secret_value()):
            raise ConfigurationError(
                "Provider credential must contain printable ASCII without spaces."
            )
        if provider == "openai":
            return {"Authorization": f"Bearer {key.get_secret_value()}"}
        return {"x-api-key": key.get_secret_value(), "anthropic-version": "2023-06-01"}

    async def _post(self, provider: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers(provider)
        root = OPENAI_ROOT if provider == "openai" else ANTHROPIC_ROOT
        for attempt in range(self.settings.api_max_retries + 1):
            response: httpx.Response | None = None
            try:
                async with self._semaphore, asyncio.timeout(self.settings.api_timeout_seconds):
                    async with self._http.stream(
                        "POST", root + path, headers=headers, json=payload
                    ) as response:
                        if response.is_success:
                            body = bytearray()
                            async for part in response.aiter_bytes():
                                if len(body) + len(part) > MAX_RESPONSE_BYTES:
                                    raise ProviderError("Provider response exceeds the byte limit.")
                                body.extend(part)
                            try:
                                value = json.loads(body)
                            except (ValueError, RecursionError):
                                raise ProviderError("Provider returned malformed JSON.") from None
                            if not isinstance(value, dict):
                                raise ProviderError("Provider returned a non-object response.")
                            return value
                if response.status_code not in RETRYABLE:
                    raise ProviderError(
                        f"{provider} rejected the request (HTTP {response.status_code}); "
                        "check credentials, model access and request settings."
                    )
            except (httpx.RequestError, TimeoutError):
                if attempt == self.settings.api_max_retries:
                    raise ProviderError(
                        f"{provider} request timed out or lost connection after bounded retries."
                    ) from None
            if attempt == self.settings.api_max_retries:
                status = response.status_code if response is not None else "network"
                raise ProviderError(f"{provider} remained unavailable (HTTP {status}); try later.")
            delay = min(8.0, self.settings.api_backoff_seconds * 2**attempt)
            if response is not None:
                try:
                    delay = max(delay, min(8.0, float(response.headers.get("Retry-After", "0"))))
                except ValueError:
                    delay = max(delay, 1.0)
            await asyncio.sleep(delay + random.uniform(0, min(delay * 0.1, 0.25)))
        raise ProviderError("Provider retry policy exhausted.")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoding = get_encoding()
        counts = await asyncio.to_thread(
            lambda: [len(encoding.encode(text, disallowed_special=())) for text in texts]
        )
        if any(not text.strip() or count > 8191 for text, count in zip(texts, counts, strict=True)):
            raise ProviderError("Embedding inputs must be nonempty and at most 8191 tokens each.")
        batch_size = self.settings.embedding_batch_size
        results: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if sum(counts[start : start + batch_size]) > 300_000:
                raise ProviderError("Embedding request exceeds the aggregate token limit.")
            value = await self._post(
                "openai",
                "/embeddings",
                {
                    "model": self.settings.embedding_model,
                    "dimensions": self.settings.embedding_dimensions,
                    "encoding_format": "float",
                    "input": batch,
                },
            )
            data = value.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ProviderError("Embedding response count does not match the input batch.")
            indexed: dict[int, list[float]] = {}
            for item in data:
                if not isinstance(item, dict) or type(item.get("index")) is not int:
                    raise ProviderError("Embedding response has an invalid item index.")
                index = item["index"]
                raw = item.get("embedding")
                if index in indexed or not 0 <= index < len(batch) or not isinstance(raw, list):
                    raise ProviderError(
                        "Embedding response indices are duplicated or out of range."
                    )
                try:
                    if any(isinstance(number, bool) for number in raw):
                        raise ValueError("Boolean vector value")
                    vector = [float(number) for number in raw]
                    norm = math.sqrt(sum(number * number for number in vector))
                except (ValueError, TypeError, OverflowError):
                    raise ProviderError(
                        "Embedding response contains invalid numeric values."
                    ) from None
                if (
                    len(vector) != self.settings.embedding_dimensions
                    or not math.isfinite(norm)
                    or norm == 0
                    or not all(math.isfinite(number) for number in vector)
                ):
                    raise ProviderError("Embedding response has invalid dimensions or norm.")
                indexed[index] = [number / norm for number in vector]
            results.extend(indexed[index] for index in range(len(batch)))
        return results

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        role: Literal["synthesis", "verification"] = "synthesis",
    ) -> dict[str, Any]:
        model = self.settings.synthesis_model
        if role == "verification" and self.settings.verification_model:
            model = self.settings.verification_model
        provider = self.settings.synthesis_provider
        wire_schema = _wire_schema(schema)
        if provider == "openai":
            value = await self._post(
                provider,
                "/responses",
                {
                    "model": model,
                    "instructions": system,
                    "input": user,
                    "store": False,
                    "max_output_tokens": self.settings.max_answer_tokens,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "mimir_result",
                            "strict": True,
                            "schema": wire_schema,
                        }
                    },
                },
            )
            if value.get("status") != "completed":
                raise ProviderError("OpenAI output was incomplete; no answer was accepted.")
            pieces: list[str] = []
            outputs = value.get("output")
            if not isinstance(outputs, list):
                raise ProviderError("OpenAI output has an invalid message container.")
            for output in outputs:
                if not isinstance(output, dict):
                    raise ProviderError("OpenAI output has an invalid message item.")
                message_type = output.get("type")
                if message_type is not None and not isinstance(message_type, str):
                    raise ProviderError("OpenAI output has an invalid message type.")
                if message_type not in {None, "message"}:
                    continue
                contents = output.get("content")
                if not isinstance(contents, list):
                    raise ProviderError("OpenAI output has an invalid content container.")
                for content in contents:
                    if not isinstance(content, dict):
                        raise ProviderError("OpenAI output has an invalid content item.")
                    if content.get("type") == "refusal":
                        raise ProviderError("OpenAI refused the synthesis request.")
                    if content.get("type") == "output_text" and isinstance(
                        content.get("text"), str
                    ):
                        pieces.append(content["text"])
        else:
            value = await self._post(
                provider,
                "/messages",
                {
                    "model": model,
                    "system": system,
                    "max_tokens": self.settings.max_answer_tokens,
                    "messages": [{"role": "user", "content": user}],
                    "output_config": {"format": {"type": "json_schema", "schema": wire_schema}},
                },
            )
            if value.get("stop_reason") != "end_turn":
                raise ProviderError("Anthropic output was refused or incomplete.")
            contents = value.get("content")
            if not isinstance(contents, list) or any(
                not isinstance(item, dict) for item in contents
            ):
                raise ProviderError("Anthropic output has an invalid content container.")
            pieces = [
                item["text"]
                for item in contents
                if item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
        try:
            decoded = json.loads("".join(pieces))
        except (ValueError, TypeError, RecursionError):
            raise ProviderError("Synthesis output did not contain valid JSON.") from None
        if not isinstance(decoded, dict):
            raise ProviderError("Synthesis output was not a JSON object.")
        return decoded
