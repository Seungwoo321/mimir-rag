from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from typing import Any, Literal, Self

import httpx

from .config import Settings
from .errors import ConfigurationError, ProviderError
from .local_embeddings import LocalEmbeddingEngine

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
        self._embeddings = LocalEmbeddingEngine(settings)
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
        key = self.settings.anthropic_api_key
        if key is None or not key.get_secret_value().strip():
            raise ConfigurationError("Set ANTHROPIC_API_KEY before Anthropic synthesis.")
        if any(not 33 <= ord(character) <= 126 for character in key.get_secret_value()):
            raise ConfigurationError(
                "Provider credential must contain printable ASCII without spaces."
            )
        return {"x-api-key": key.get_secret_value(), "anthropic-version": "2023-06-01"}

    async def _post(self, provider: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers(provider)
        root = ANTHROPIC_ROOT
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

    async def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        return await self._embeddings.embed(texts, purpose=purpose)

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
        if provider == "claude-code":
            from .claude_code import ClaudeCodeClient

            return await ClaudeCodeClient(self.settings).complete_json(
                system, user, schema, role=role
            )
        wire_schema = _wire_schema(schema)
        if provider == "agent":
            raise ConfigurationError(
                "Agent synthesis requires the host agent; use the evidence retrieval command."
            )
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
        if not isinstance(contents, list) or any(not isinstance(item, dict) for item in contents):
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
