from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from mimir_rag.config import Settings
from mimir_rag.generator import Generator
from mimir_rag.models import Chunk, SearchHit
from mimir_rag.providers import ProviderClient
from mimir_rag.vector_store import VectorStore


def draft() -> dict[str, Any]:
    return {
        "answerable": True,
        "claims": [
            {
                "id": "c1",
                "text": "The source says observations can change actions.",
                "evidence": [{"source_id": "s1", "quote": "Observed outcomes"}],
            }
        ],
        "connections": [],
    }


def verdict() -> dict[str, Any]:
    return {
        "approved": True,
        "claim_verdicts": [{"id": "c1", "supported": True}],
        "connection_verdicts": [],
    }


class EvidenceStore:
    async def has_chunks(self) -> bool:
        return True

    def __init__(self, path: Path) -> None:
        text = "Observed outcomes can inform revisions to actions."
        self.hit = SearchHit(
            chunk=Chunk(
                id="s1",
                document_id="d1",
                ordinal=0,
                text=text,
                title="Synthetic Book",
                section="Feedback",
                char_start=0,
                char_end=len(text),
                tokens=10,
                tags=["Core Theory"],
                concepts=[],
            ),
            score=0.03,
            dense_score=0.8,
            lexical_rank=1,
            source_uri=(path / "synthetic.txt").as_uri(),
        )

    async def search(
        self, question: str, vector: list[float], top_k: int | None = None
    ) -> list[SearchHit]:
        return [self.hit]


def envelope(provider: str, value: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(value)
    if provider == "openai":
        return {
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": text}]}],
        }
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


@pytest.mark.parametrize("role", ["synthesis", "verification"])
@pytest.mark.parametrize(
    "provider,container", [("openai", "output"), ("openai", "content"), ("anthropic", "content")]
)
async def test_null_provider_containers_produce_safe_abstention(
    tmp_path: Path, role: str, provider: str, container: str
) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        calls += 1
        phase = "synthesis" if calls == 1 else "verification"
        value = envelope(provider, draft() if calls == 1 else verdict())
        if phase == role:
            if provider == "openai" and container == "content":
                value["output"] = [{"content": None}]
            else:
                value[container] = None
        return httpx.Response(200, json=value)

    settings = Settings(
        db_path=tmp_path / "library.sqlite3",
        embedding_dimensions=2,
        synthesis_provider=provider,
        openai_api_key="synthetic-key",
        anthropic_api_key="synthetic-key",
        api_max_retries=0,
    )
    async with ProviderClient(settings, transport=httpx.MockTransport(respond)) as client:
        answer = await Generator(settings, cast(VectorStore, EvidenceStore(tmp_path)), client).ask(
            "How do observations affect actions?"
        )
    assert answer.abstained
    assert answer.reason == f"{role}_provider_failure"
    assert answer.source_ids == []


class LargeResponse(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.bytes_read = 0

    async def __aiter__(self):
        prefix = b'{"status":"completed","padding":"'
        self.bytes_read += len(prefix)
        yield prefix
        for _ in range(17):
            piece = b"x" * (1024 * 1024)
            self.bytes_read += len(piece)
            yield piece
        suffix = ',"output":' + json.dumps(envelope("openai", draft())["output"]) + "}"
        piece = ('"' + suffix).encode()
        self.bytes_read += len(piece)
        yield piece


async def test_oversized_provider_body_abstains_before_unbounded_buffering(tmp_path: Path) -> None:
    body = LargeResponse()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=body)

    settings = Settings(
        db_path=tmp_path / "library.sqlite3",
        embedding_dimensions=2,
        openai_api_key="synthetic-key",
        api_max_retries=0,
    )
    async with ProviderClient(settings, transport=httpx.MockTransport(respond)) as client:
        answer = await Generator(settings, cast(VectorStore, EvidenceStore(tmp_path)), client).ask(
            "How do observations affect actions?"
        )
    assert answer.abstained and answer.reason == "synthesis_provider_failure"
    assert body.bytes_read <= 17 * 1024 * 1024


class ControlledSynthesis:
    def __init__(self, proposal: dict[str, Any], review: dict[str, Any]) -> None:
        self.proposal, self.review = proposal, review
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]

    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, role: str = "synthesis"
    ) -> dict[str, Any]:
        self.calls += 1
        return self.proposal if role == "synthesis" else self.review


@pytest.mark.parametrize("invalid", ["\ud800", "\udfff"])
async def test_lone_surrogate_model_prose_cannot_be_published(tmp_path: Path, invalid: str) -> None:
    proposal = draft()
    proposal["claims"][0]["text"] += invalid
    client = ControlledSynthesis(proposal, verdict())
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=2)
    answer = await Generator(
        settings, cast(VectorStore, EvidenceStore(tmp_path)), cast(ProviderClient, client)
    ).ask("How do observations affect actions?")
    assert answer.abstained
    assert answer.source_ids == []


@pytest.mark.parametrize("invalid", [None, [], {"approved": float("nan")}, {"answerable": 1}])
async def test_malformed_structured_output_abstains(tmp_path: Path, invalid: Any) -> None:
    client = ControlledSynthesis(invalid, verdict())
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=2)
    answer = await Generator(
        settings, cast(VectorStore, EvidenceStore(tmp_path)), cast(ProviderClient, client)
    ).ask("How do observations affect actions?")
    assert answer.abstained and answer.reason == "invalid_synthesis"
    assert client.calls == 1


async def test_review_rejects_missing_negative_or_extra_concept_verdicts(tmp_path: Path) -> None:
    source = EvidenceStore(tmp_path)
    source.hit = source.hit.model_copy(
        update={"chunk": source.hit.chunk.model_copy(update={"concepts": ["outcomes", "actions"]})}
    )
    proposal = draft()
    proposal["connections"] = [
        {
            "id": "map1",
            "concept_a": "outcomes",
            "concept_b": "actions",
            "relation": "co_occurs",
            "text": "The passage mentions both concepts.",
            "evidence": [{"source_id": "s1", "quote": "Observed outcomes"}],
        }
    ]
    for entries in (
        [],
        [{"id": "map1", "supported": False}],
        [{"id": "map1", "supported": True}, {"id": "unknown", "supported": True}],
    ):
        review = verdict()
        review["connection_verdicts"] = entries
        client = ControlledSynthesis(proposal, review)
        settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=2)
        answer = await Generator(
            settings, cast(VectorStore, source), cast(ProviderClient, client)
        ).ask("How do observations affect actions?")
        assert answer.abstained and answer.reason == "review_rejected"
