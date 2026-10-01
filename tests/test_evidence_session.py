from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from mimir_rag.config import Settings
from mimir_rag.errors import GroundingError
from mimir_rag.evidence_session import EvidenceSessions, read_private_json
from mimir_rag.generator import Generator
from mimir_rag.models import Chunk, Document
from mimir_rag.providers import ProviderClient
from mimir_rag.vector_store import VectorStore


class LocalStub:
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        assert purpose == "query"
        return [[1.0, 0.0] for _ in texts]

    async def complete_json(self, *args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("Host-agent evidence flow must not call a synthesis API.")


def private_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


async def corpus(tmp_path: Path) -> tuple[EvidenceSessions, VectorStore, dict[str, Any]]:
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=2)
    store = VectorStore(settings)
    text = "Feedback adjusts actions using observed outcomes."
    document = Document(
        id="book",
        source_uri=(tmp_path / "book.md").as_uri(),
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        generation="one",
        title="Synthetic Handbook",
        rights="Synthetic test fixture",
        metadata_origin="user",
        ingest_signature="fixture",
    )
    chunk = Chunk(
        id="chunk",
        document_id="book",
        ordinal=0,
        text=text,
        title=document.title,
        section="Feedback",
        char_start=0,
        char_end=len(text),
        tokens=10,
        tags=["Core Theory"],
        concepts=["feedback", "actions"],
        line_start=1,
        line_end=1,
    )
    await store.upsert_document(document, [chunk], [[1.0, 0.0]])
    sessions = EvidenceSessions(Generator(settings, store, cast(ProviderClient, LocalStub())))
    envelope = await sessions.retrieve("How does feedback affect actions?")
    return sessions, store, envelope


def proposal() -> dict[str, Any]:
    return {
        "answerable": True,
        "claims": [
            {
                "id": "c1",
                "text": "The handbook links feedback with revising actions.",
                "evidence": [{"source_id": "chunk", "quote": "Feedback adjusts actions"}],
            }
        ],
        "connections": [],
    }


async def files(
    tmp_path: Path, sessions: EvidenceSessions, envelope: dict[str, Any]
) -> tuple[Path, Path]:
    draft = private_json(tmp_path / "proposal.json", proposal())
    context = await sessions.review_context(
        envelope["session_id"], envelope["evidence_hash"], draft
    )
    review = private_json(
        tmp_path / "review.json",
        {
            "evidence_hash": context["evidence_hash"],
            "proposal_hash": context["proposal_hash"],
            "review": {
                "approved": True,
                "claim_verdicts": [{"id": "c1", "supported": True}],
                "connection_verdicts": [],
            },
        },
    )
    return draft, review


async def test_real_store_host_flow_citation_first_and_private_session(tmp_path: Path) -> None:
    sessions, _, envelope = await corpus(tmp_path)
    assert envelope["status"] == "awaiting_proposal"
    assert envelope["sources"][0]["source_uri"].endswith("book.md")
    assert "prior knowledge is not evidence" in envelope["synthesis_system_prompt"]
    assert stat.S_IMODE(sessions.directory.stat().st_mode) == 0o700
    session_path = sessions.directory / f"{envelope['session_id']}.json"
    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
    draft, review = await files(tmp_path, sessions, envelope)
    answer = await sessions.verify(envelope["session_id"], envelope["evidence_hash"], draft, review)
    assert not answer.abstained and answer.source_ids == ["chunk"]
    assert answer.markdown.startswith("- [Synthetic Handbook")
    assert answer.markdown.index(")") < answer.markdown.index("The handbook")


@pytest.mark.parametrize("mutation", ["deleted", "chunk", "uri", "hash", "binding", "expired"])
async def test_stale_or_tampered_sources_never_publish(tmp_path: Path, mutation: str) -> None:
    sessions, store, envelope = await corpus(tmp_path)
    draft, review = await files(tmp_path, sessions, envelope)
    if mutation == "deleted":
        await store.delete_document("book")
    elif mutation in {"chunk", "uri"}:
        import sqlite3

        with sqlite3.connect(store.path) as connection:
            if mutation == "chunk":
                connection.execute("UPDATE chunks SET title='Changed' WHERE id='chunk'")
            else:
                connection.execute(
                    "UPDATE documents SET source_uri='file:///changed' WHERE id='book'"
                )
    else:
        path = sessions.directory / f"{envelope['session_id']}.json"
        value = read_private_json(path)
        value[{"hash": "question", "binding": "binding", "expired": "created_at"}[mutation]] = (
            0 if mutation == "expired" else "changed"
        )
        private_json(path, value)
        if mutation != "hash":
            envelope["evidence_hash"] = hashlib.sha256(
                json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
    with pytest.raises(GroundingError):
        await sessions.verify(envelope["session_id"], envelope["evidence_hash"], draft, review)


@pytest.mark.parametrize(
    "mutation", ["proposal_hash", "evidence_hash", "missing", "negative", "extra"]
)
async def test_review_binding_and_full_verdict_coverage(tmp_path: Path, mutation: str) -> None:
    sessions, _, envelope = await corpus(tmp_path)
    draft, review = await files(tmp_path, sessions, envelope)
    value = read_private_json(review)
    if mutation.endswith("hash"):
        value[mutation] = "0" * 64
    elif mutation == "missing":
        value["review"]["claim_verdicts"] = []
    elif mutation == "negative":
        value["review"]["claim_verdicts"][0]["supported"] = False
    else:
        value["review"]["claim_verdicts"].append({"id": "unknown", "supported": True})
    private_json(review, value)
    if mutation.endswith("hash"):
        with pytest.raises(GroundingError, match="bound"):
            await sessions.verify(envelope["session_id"], envelope["evidence_hash"], draft, review)
    else:
        answer = await sessions.verify(
            envelope["session_id"], envelope["evidence_hash"], draft, review
        )
        assert answer.abstained and answer.reason == "review_rejected"


@pytest.mark.parametrize(
    "mutation", ["fake_quote", "source", "duplicate_id", "concept", "unanswerable"]
)
async def test_proposal_validation_precedes_host_review(tmp_path: Path, mutation: str) -> None:
    sessions, _, envelope = await corpus(tmp_path)
    value = proposal()
    if mutation == "fake_quote":
        value["claims"][0]["evidence"][0]["quote"] = "absent quote"
    elif mutation == "source":
        value["claims"][0]["evidence"][0]["source_id"] = "invented"
    elif mutation == "duplicate_id":
        value["claims"].append(copy.deepcopy(value["claims"][0]))
    elif mutation == "concept":
        value["connections"] = [
            {
                "id": "map",
                "concept_a": "invented",
                "concept_b": "actions",
                "relation": "co_occurs",
                "text": "A co-occurrence.",
                "evidence": copy.deepcopy(value["claims"][0]["evidence"]),
            }
        ]
    else:
        value["answerable"] = False
    draft = private_json(tmp_path / "proposal.json", value)
    with pytest.raises(GroundingError):
        await sessions.review_context(envelope["session_id"], envelope["evidence_hash"], draft)


@pytest.mark.parametrize(
    "unsafe",
    ["public", "symlink", "oversize", "invalid", "duplicate", "nonfinite", "surrogate", "deep"],
)
def test_private_input_bounds(tmp_path: Path, unsafe: str) -> None:
    path = private_json(tmp_path / "input.json", {})
    if unsafe == "public":
        path.chmod(0o644)
    elif unsafe == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif unsafe == "oversize":
        path.write_bytes(b" " * 4_000_001)
    elif unsafe == "duplicate":
        path.write_bytes(b'{"x":1,"x":2}')
    elif unsafe == "nonfinite":
        path.write_bytes(b'{"x":NaN}')
    elif unsafe == "surrogate":
        path.write_text('{"x":"\\ud800"}')
    elif unsafe == "deep":
        path.write_bytes(b"[" * 2000 + b"]" * 2000)
    else:
        path.write_bytes(b"{")
    with pytest.raises(GroundingError):
        read_private_json(path)


async def test_session_identity_prevents_path_traversal(tmp_path: Path) -> None:
    sessions, _, envelope = await corpus(tmp_path)
    with pytest.raises(GroundingError, match="identity"):
        await sessions.review_context("../proposal", envelope["evidence_hash"], tmp_path / "x")
