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
        anthropic_api_key=SecretStr("test-key"),
        synthesis_provider="anthropic",
        api_backoff_seconds=0,
        **values,
    )


def successful() -> dict[str, Any]:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"ok":true}'}]}


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
        return httpx.Response(200, json=successful())

    async with ProviderClient(
        settings(api_max_retries=2), transport=httpx.MockTransport(handle)
    ) as client:
        assert await client.complete_json("system", "user", {}) == {"ok": True}
    assert attempts == 3


@pytest.mark.parametrize("status,expected", [(401, 1), (503, 2)])
async def test_exhaustion_and_auth_do_not_leak_response(status: int, expected: int) -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": "source-content-and-secret"})

    async with ProviderClient(
        settings(api_max_retries=1), transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ProviderError) as error:
            await client.complete_json("system", "user", {})
    assert calls == expected
    assert "source-content-and-secret" not in str(error.value)


@pytest.mark.parametrize(
    "configuration",
    [Settings(synthesis_provider="agent"), Settings(synthesis_provider="anthropic")],
)
async def test_host_agent_or_missing_key_fails_before_http(configuration: Settings) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Unexpected cloud call")

    async with ProviderClient(configuration, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ConfigurationError):
            await client.complete_json("system", "user", {})


async def test_anthropic_structured_review_uses_fixed_system_and_model() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.host == "api.anthropic.com"
        assert request.url.path == "/v1/messages"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["system"] == "fixed-system"
        assert body["model"] == "claude-review-model"
        return httpx.Response(200, json=successful())

    async with ProviderClient(
        settings(verification_model="claude-review-model"), transport=httpx.MockTransport(handle)
    ) as client:
        assert await client.complete_json(
            "fixed-system", "data", {"type": "object"}, role="verification"
        ) == {"ok": True}


@pytest.mark.parametrize(
    "response",
    [
        {"stop_reason": "max_tokens", "content": []},
        {"stop_reason": "refusal", "content": []},
        {"stop_reason": "end_turn", "content": {}},
        {"stop_reason": "end_turn", "content": [True]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "invalid"}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "[]"}]},
    ],
)
async def test_refused_incomplete_or_malformed_synthesis_fails(response: dict[str, Any]) -> None:
    async with ProviderClient(
        settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
    ) as client:
        with pytest.raises(ProviderError):
            await client.complete_json("system", "user", {})


@pytest.mark.parametrize("key", [" ", "invalid key", "nonascii\u00e9", "bad\nkey"])
async def test_invalid_credentials_fail_without_network(key: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Unexpected request")

    configuration = Settings(synthesis_provider="anthropic", anthropic_api_key=SecretStr(key))
    async with ProviderClient(configuration, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ConfigurationError):
            await client.complete_json("s", "u", {})
