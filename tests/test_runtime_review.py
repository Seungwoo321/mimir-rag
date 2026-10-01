from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from mimir_rag import main as cli
from mimir_rag.config import Settings
from mimir_rag.errors import ConfigurationError, ProviderError
from mimir_rag.providers import MAX_RESPONSE_BYTES, ProviderClient

SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}},
    "required": ["accepted"],
    "additionalProperties": False,
}


def configuration(**changes: Any) -> Settings:
    return Settings(
        openai_api_key=SecretStr("synthetic-openai-key"),
        anthropic_api_key=SecretStr("synthetic-anthropic-key"),
        embedding_dimensions=2,
        api_max_retries=1,
        api_backoff_seconds=0,
        **changes,
    )


def accepted_response(provider: str) -> dict[str, Any]:
    content = [
        {"type": "output_text" if provider == "openai" else "text", "text": '{"accepted":true}'}
    ]
    if provider == "openai":
        return {"status": "completed", "output": [{"type": "message", "content": content}]}
    return {"stop_reason": "end_turn", "content": content}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("container", [None, 17, "message", {"text": "data"}])
async def test_synthesis_container_failures_are_normalized(provider: str, container: Any) -> None:
    value = accepted_response(provider)
    if provider == "openai":
        value["output"] = container
    else:
        value["content"] = container
    async with ProviderClient(
        configuration(synthesis_provider=provider),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=value)),
    ) as client:
        with pytest.raises(ProviderError):
            await client.complete_json("fixed system", "untrusted data", SCHEMA)


@pytest.mark.parametrize("container", [None, 5, "contents", {"type": "output_text"}])
async def test_openai_nested_content_container_failures_are_normalized(container: Any) -> None:
    value = accepted_response("openai")
    value["output"][0]["content"] = container
    async with ProviderClient(
        configuration(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=value)),
    ) as client:
        with pytest.raises(ProviderError):
            await client.complete_json("fixed system", "untrusted data", SCHEMA)


@pytest.mark.parametrize("message_type", [[], {}, 7, True])
async def test_openai_malformed_message_type_fails_closed(message_type: Any) -> None:
    value = accepted_response("openai")
    value["output"][0]["type"] = message_type
    async with ProviderClient(
        configuration(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=value)),
    ) as client:
        with pytest.raises(ProviderError):
            await client.complete_json("fixed system", "untrusted data", SCHEMA)


@pytest.mark.parametrize(
    "error_class",
    [httpx.RemoteProtocolError, httpx.DecodingError, httpx.UnsupportedProtocol, httpx.ConnectError],
)
async def test_all_request_transport_failures_are_bounded_and_secret_safe(error_class: Any) -> None:
    calls = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_class("synthetic-secret-and-document-text", request=request)

    async with ProviderClient(configuration(), transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(ProviderError) as error:
            await client.embed(["synthetic source"])
    assert calls == 2
    assert "synthetic-secret-and-document-text" not in str(error.value)
    assert error.value.__suppress_context__


async def test_remote_protocol_failure_can_recover_within_retry_budget() -> None:
    calls = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("broken HTTP framing", request=request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    async with ProviderClient(configuration(), transport=httpx.MockTransport(flaky)) as client:
        assert await client.embed(["synthetic source"]) == [[1.0, 0.0]]
    assert calls == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "suffix", ["\nsecret", "\rsecret", "\x00secret", "\u00e9secret", " secret"]
)
async def test_invalid_credentials_fail_before_http_without_secret_disclosure(
    provider: str, suffix: str
) -> None:
    called = False
    secret = "synthetic-private-key" + suffix

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=accepted_response(provider))

    values = {
        "synthesis_provider": provider,
        "openai_api_key": SecretStr(secret),
        "anthropic_api_key": SecretStr(secret),
        "embedding_dimensions": 2,
    }
    async with ProviderClient(
        Settings.model_validate(values), transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ConfigurationError) as error:
            await client.complete_json("fixed system", "source", SCHEMA)
    assert not called
    assert "synthetic-private-key" not in str(error.value)


class OversizedStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        for _ in range(MAX_RESPONSE_BYTES // 4096 + 1):
            yield b"x" * 4096

    async def aclose(self) -> None:
        self.closed = True


async def test_oversized_success_body_is_rejected_and_stream_closed() -> None:
    stream = OversizedStream()
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=stream)

    async with ProviderClient(configuration(), transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ProviderError, match="byte limit"):
            await client.embed(["synthetic source"])
    assert calls == 1
    assert stream.closed


async def test_entire_attempt_deadline_covers_slow_response_delivery() -> None:
    calls = 0

    async def delayed(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    async with ProviderClient(
        configuration(api_timeout_seconds=0.001), transport=httpx.MockTransport(delayed)
    ) as client:
        with pytest.raises(ProviderError):
            await client.embed(["synthetic source"])
    assert calls == 2


async def test_embedding_budget_violation_makes_no_provider_calls() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with ProviderClient(
        configuration(embedding_batch_size=64), transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ProviderError, match="8191"):
            await client.embed([" word" * 8192])
        with pytest.raises(ProviderError, match="aggregate"):
            await client.embed([" word" * 5000] * 64)
    assert calls == 0


async def test_schema_adaptation_preserves_property_and_definition_names() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"$ref": "#/$defs/title"},
            "pattern": {"type": "string", "minLength": 1},
        },
        "required": ["title", "pattern"],
        "additionalProperties": False,
        "$defs": {"title": {"type": "string", "maxLength": 20}},
    }

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wired = body["text"]["format"]["schema"]
        assert set(wired["properties"]) == {"title", "pattern"}
        assert "title" in wired["$defs"]
        assert "minLength" not in wired["properties"]["pattern"]
        return httpx.Response(200, json=accepted_response("openai"))

    async with ProviderClient(configuration(), transport=httpx.MockTransport(handle)) as client:
        await client.complete_json("fixed system", "source", schema)


def test_literal_option_like_question_and_aliases_are_preserved(monkeypatch) -> None:
    captured: list[argparse.Namespace] = []

    async def capture(args: argparse.Namespace) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(cli, "_run", capture)
    question = "--help /ask /ingest `echo private` $(echo private)"
    assert cli.main(["/ask", "--json", "--", question]) == 0
    assert captured[0].question == [question]
    assert captured[0].command == "ask"


def test_host_commands_preserve_cwd_and_use_question_option_terminator() -> None:
    root = Path(__file__).resolve().parents[1]
    ask = (root / "commands/ask.md").read_text()
    ingest = (root / "commands/ingest.md").read_text()
    assert "--directory" not in ask and "--directory" not in ingest
    assert "uv run --project" in ask and "uv run --project" in ingest
    assert "mimir-rag ask --json --" in ask
