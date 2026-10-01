import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from mimir_rag.config import Settings
from mimir_rag.errors import ConfigurationError, ProviderError
from mimir_rag.providers import ProviderClient


def settings(**values: Any) -> Settings:
    return Settings(
        openai_api_key=SecretStr("test-key"),
        embedding_dimensions=2,
        api_backoff_seconds=0,
        **values,
    )


async def test_embeddings_preserve_input_order_and_normalize() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["alpha", "beta"]
        assert body["dimensions"] == 2
        assert body["model"] == "text-embedding-3-small"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 9]},
                    {"index": 0, "embedding": [3, 4]},
                ]
            },
        )

    async with ProviderClient(settings(), transport=httpx.MockTransport(handle)) as client:
        assert await client.embed(["alpha", "beta"]) == [[0.6, 0.8], [0.0, 1.0]]


@pytest.mark.parametrize("mode", ["timeout", "rate-limit", "server-error"])
async def test_bounded_transient_retry(mode: str) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            if mode == "timeout":
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            return httpx.Response(429 if mode == "rate-limit" else 503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    async with ProviderClient(
        settings(api_max_retries=2), transport=httpx.MockTransport(handle)
    ) as client:
        assert await client.embed(["alpha"]) == [[1.0, 0.0]]
    assert attempts == 3


async def test_exhaustion_and_nonretryable_auth_do_not_leak_response() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "source-content-and-secret"})

    async with ProviderClient(settings(), transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ProviderError) as error:
            await client.embed(["alpha"])
    assert calls == 1
    assert "source-content-and-secret" not in str(error.value)
    calls = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with ProviderClient(
        settings(api_max_retries=1), transport=httpx.MockTransport(unavailable)
    ) as client:
        with pytest.raises(ProviderError):
            await client.embed(["alpha"])
    assert calls == 2


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": [1]}],
        [{"index": 2, "embedding": [1, 0]}],
        [{"index": 0, "embedding": [True, 0]}],
    ],
)
async def test_malformed_vectors_are_rejected(data: list[dict[str, Any]]) -> None:
    async with ProviderClient(
        settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": data})),
    ) as client:
        with pytest.raises(ProviderError):
            await client.embed(["alpha"])


async def test_missing_key_fails_before_http_request() -> None:
    called = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    async with ProviderClient(Settings(), transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ConfigurationError):
            await client.embed(["alpha"])
    assert not called


async def test_openai_structured_output_has_storage_disabled_and_review_model() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["store"] is False
        assert body["model"] == "review-model"
        assert body["instructions"] == "fixed-system"
        assert body["text"]["format"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": '{"approved":true}'},
                        ]
                    }
                ],
            },
        )

    async with ProviderClient(
        settings(verification_model="review-model"), transport=httpx.MockTransport(handle)
    ) as client:
        assert await client.complete_json(
            "fixed-system", "data", {"type": "object"}, role="verification"
        ) == {"approved": True}


async def test_anthropic_uses_messages_structured_format() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["system"] == "fixed-system"
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [
                    {"type": "text", "text": '{"answerable":false}'},
                ],
            },
        )

    configuration = settings(
        synthesis_provider="anthropic",
        synthesis_model="claude-haiku-4-5-20251001",
        anthropic_api_key=SecretStr("test-key"),
    )
    async with ProviderClient(configuration, transport=httpx.MockTransport(handle)) as client:
        assert await client.complete_json("fixed-system", "data", {"type": "object"}) == {
            "answerable": False
        }


@pytest.mark.parametrize(
    "response",
    [
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": [{"content": [{"type": "refusal"}]}]},
        {
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": "invalid"}]}],
        },
    ],
)
async def test_refused_incomplete_or_malformed_synthesis_fails(response: dict[str, Any]) -> None:
    async with ProviderClient(
        settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
    ) as client:
        with pytest.raises(ProviderError):
            await client.complete_json("system", "user", {"type": "object"})
