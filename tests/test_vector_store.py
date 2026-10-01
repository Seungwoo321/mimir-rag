from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import stat
import threading
from pathlib import Path

import numpy as np
import pytest

from mimir_rag.config import Settings
from mimir_rag.errors import EmbeddingSpaceError, StoreError
from mimir_rag.models import Chunk, Document
from mimir_rag.vector_store import VectorStore


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    return Settings.model_validate(
        {
            "db_path": tmp_path / "private-library" / "library.sqlite3",
            "embedding_dimensions": 3,
            "dense_batch_size": 1,
            **overrides,
        }
    )


def source(name: str, texts: list[str], generation: str = "one") -> tuple[Document, list[Chunk]]:
    document = Document(
        id=name,
        source_uri=f"file:///synthetic/{name}.md",
        content_hash=hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        generation=generation,
        title=f"Synthetic {name}",
        author=None,
        rights="Synthetic test fixture",
        metadata_origin="user override",
        ingest_signature="test",
    )
    chunks = [
        Chunk(
            id=f"{name}-{generation}-{index}",
            document_id=name,
            ordinal=index,
            text=text,
            title=document.title,
            section=f"Section {index}",
            char_start=0,
            char_end=len(text),
            tokens=len(text.split()),
            tags=["Core Theory"],
            concepts=["Feedback", "Control Theory"],
            line_start=index + 1,
            line_end=index + 1,
        )
        for index, text in enumerate(texts)
    ]
    return document, chunks


async def test_hybrid_ranking_provenance_and_normalized_storage(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    document, chunks = source("ranking", ["A semantic passage", "Needle lexical evidence", "Other"])
    assert await store.upsert_document(document, chunks, [[9, 0, 0], [4, 3, 0], [0, 0, 5]])
    hits = await store.search('needle" OR title:("irrelevant")', [2, 0, 0], top_k=3)
    assert hits[0].chunk == chunks[1]
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_score == pytest.approx(0.8)
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0].source_uri == document.source_uri
    assert await store.get_document(document.id) == document
    with sqlite3.connect(settings.db_path) as conn:
        blob = conn.execute("SELECT embedding FROM chunks WHERE id=?", (chunks[0].id,)).fetchone()[
            0
        ]
        assert len(blob) == 12
        assert np.frombuffer(blob, dtype="<f4").tolist() == [1.0, 0.0, 0.0]
    assert [hit.chunk.id for hit in hits] == [
        hit.chunk.id for hit in await store.search("needle", [2, 0, 0], top_k=3)
    ]


async def test_identical_generation_is_noop_and_replacement_refreshes_fts(tmp_path: Path) -> None:
    store = VectorStore(settings_for(tmp_path))
    old, chunks = source("replace", ["oldneedle"])
    assert await store.upsert_document(old, chunks, [[1, 0, 0]])
    assert not await store.upsert_document(old, chunks, [[1, 0, 0]])
    new, replacement = source("replace", ["newneedle", "new concept"], "two")
    assert await store.upsert_document(new, replacement, [[1, 0, 0], [0, 1, 0]])
    assert await store.document_chunk_count(old.id) == 2
    assert all(hit.lexical_rank is None for hit in await store.search("oldneedle", [1, 0, 0]))
    assert (await store.search("newneedle", [1, 0, 0]))[0].lexical_rank == 1


async def test_mid_transaction_constraint_failure_rolls_back_every_index(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    old, chunks = source("atomic", ["oldneedle"])
    foreign, foreign_chunks = source("foreign", ["otherword"])
    await store.upsert_document(old, chunks, [[1, 0, 0]])
    await store.upsert_document(foreign, foreign_chunks, [[0, 1, 0]])
    new, replacement = source("atomic", ["newword"], "two")
    conflicting = replacement[0].model_copy(update={"id": foreign_chunks[0].id})
    with pytest.raises(StoreError) as error:
        await store.upsert_document(new, [conflicting], [[1, 0, 0]])
    assert "foreign" not in str(error.value)
    assert await store.get_document(old.id) == old
    assert await store.document_chunk_count(old.id) == 1
    assert (await store.search("oldneedle", [1, 0, 0]))[0].chunk == chunks[0]
    with sqlite3.connect(settings.db_path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT count(*) FROM concept_mentions").fetchone()[0] == 4


async def test_concurrent_initialization_and_complete_source_replacement(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    stores = [VectorStore(settings) for _ in range(4)]
    await asyncio.gather(*(store.initialize() for store in stores))
    jobs = [
        source("shared", [f"generation {index}", f"tail {index}"], str(index)) for index in range(4)
    ]
    assert all(
        await asyncio.gather(
            *(
                store.upsert_document(document, chunks, [[1, 0, 0], [0, 1, 0]])
                for store, (document, chunks) in zip(stores, jobs, strict=True)
            )
        )
    )
    document = await stores[0].get_document("shared")
    assert document is not None
    hits = await stores[0].search("generation", [1, 0, 0], top_k=10)
    assert len(hits) == 2
    assert all(hit.chunk.id.startswith(f"shared-{document.generation}-") for hit in hits)


async def test_reader_uses_one_snapshot_during_writer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VectorStore(settings_for(tmp_path))
    old, chunks = source("snapshot", ["oldneedle first", "oldneedle second"])
    await store.upsert_document(old, chunks, [[1, 0, 0], [0, 1, 0]])
    entered, proceed = threading.Event(), threading.Event()
    original = store._decode

    def paused(blob: bytes) -> np.ndarray:
        entered.set()
        assert proceed.wait(timeout=5)
        return original(blob)

    monkeypatch.setattr(store, "_decode", paused)
    pending = asyncio.create_task(store.search("oldneedle", [1, 0, 0]))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        new, replacement = source("snapshot", ["newneedle"], "two")
        await store.upsert_document(new, replacement, [[1, 0, 0]])
    finally:
        proceed.set()
    hits = await pending
    assert {hit.chunk.id for hit in hits} == {chunk.id for chunk in chunks}
    assert all(hit.lexical_rank is not None for hit in hits)


@pytest.mark.parametrize(
    "embedding_model,dimensions", [("another-model", 3), ("text-embedding-3-small", 2)]
)
async def test_embedding_fingerprint_rejects_drift_without_changes(
    tmp_path: Path, embedding_model: str, dimensions: int
) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    document, chunks = source("drift", ["licensed synthetic text"])
    await store.upsert_document(document, chunks, [[1, 0, 0]])
    before = settings.db_path.read_bytes()
    altered = settings.model_copy(
        update={"embedding_model": embedding_model, "embedding_dimensions": dimensions}
    )
    with pytest.raises(EmbeddingSpaceError, match="new MIMIR_DB_PATH"):
        await VectorStore(altered).initialize()
    assert settings.db_path.read_bytes() == before


@pytest.mark.parametrize("invalid", [[0, 0, 0], [1, 2], [float("nan"), 0, 1], [0, float("inf"), 1]])
async def test_invalid_vectors_do_not_replace_document(
    tmp_path: Path, invalid: list[float]
) -> None:
    store = VectorStore(settings_for(tmp_path))
    old, chunks = source("invalid", ["oldsource"])
    await store.upsert_document(old, chunks, [[1, 0, 0]])
    new, replacement = source("invalid", ["replacement"], "two")
    with pytest.raises(EmbeddingSpaceError):
        await store.upsert_document(new, replacement, [invalid])
    assert await store.get_document(old.id) == old
    with pytest.raises(EmbeddingSpaceError):
        await store.search("question", invalid)


async def test_extreme_finite_vectors_normalize_without_overflow(tmp_path: Path) -> None:
    store = VectorStore(settings_for(tmp_path))
    document, chunks = source("large", ["finite numbers"])
    await store.upsert_document(document, chunks, [[1e308, 1e308, 0]])
    assert (await store.search("finite", [1, 1, 0]))[0].dense_score == pytest.approx(1)


async def test_corrupt_persisted_vector_is_rejected(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    document, chunks = source("corrupt-vector", ["synthetic source"])
    await store.upsert_document(document, chunks, [[1, 0, 0]])
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("UPDATE chunks SET embedding=?", (b"broken",))
    with pytest.raises(EmbeddingSpaceError, match="corrupt"):
        await store.search("source", [1, 0, 0])


@pytest.mark.parametrize("kind", ["future", "missing-meta", "corrupt", "incomplete"])
async def test_unknown_database_is_preserved_before_wal_or_chmod(tmp_path: Path, kind: str) -> None:
    settings = settings_for(tmp_path)
    settings.db_path.parent.mkdir(mode=0o700)
    if kind in {"future", "missing-meta"}:
        await VectorStore(settings).initialize()
        with sqlite3.connect(settings.db_path) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
            if kind == "future":
                conn.execute("UPDATE library_meta SET value='999' WHERE key='schema_version'")
            else:
                conn.execute("DELETE FROM library_meta WHERE key='embedding_fingerprint'")
    elif kind == "incomplete":
        with sqlite3.connect(settings.db_path) as conn:
            conn.execute("CREATE TABLE documents(id TEXT)")
    else:
        settings.db_path.write_bytes(b"Not a SQLite database; preserve this source")
    settings.db_path.chmod(0o640)
    before = settings.db_path.read_bytes()
    mode = stat.S_IMODE(settings.db_path.stat().st_mode)
    with pytest.raises(StoreError):
        await VectorStore(settings).initialize()
    assert settings.db_path.read_bytes() == before
    assert stat.S_IMODE(settings.db_path.stat().st_mode) == mode


async def test_capacity_check_serializes_concurrent_writers(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, max_chunks=3)
    stores = [VectorStore(settings), VectorStore(settings)]
    await stores[0].initialize()
    jobs = [source(name, ["first", "second"]) for name in ["left", "right"]]
    results = await asyncio.gather(
        *(
            store.upsert_document(document, chunks, [[1, 0, 0], [0, 1, 0]])
            for store, (document, chunks) in zip(stores, jobs, strict=True)
        ),
        return_exceptions=True,
    )
    assert results.count(True) == 1
    assert sum(isinstance(result, StoreError) for result in results) == 1
    assert len(await stores[0].list_documents()) == 1


async def test_delete_removes_dense_lexical_and_concept_references(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    document, chunks = source("remove", ["needle feedback"])
    await store.upsert_document(document, chunks, [[1, 0, 0]])
    graph = await store.concept_graph(document.id)
    assert {edge["type"] for edge in graph["edges"]} == {"mentions", "co_occurrence"}
    assert all(edge.get("chunk_id", chunks[0].id) == chunks[0].id for edge in graph["edges"])
    assert await store.delete_document(document.id)
    assert not await store.delete_document(document.id)
    assert await store.list_documents() == []
    assert await store.search("needle", [1, 0, 0]) == []
    assert await store.concept_graph() == {"nodes": [], "edges": []}
    with sqlite3.connect(settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM concept_mentions").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'needle'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and permissions contract")
async def test_private_directory_creation_never_chmods_arbitrary_ancestors(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    settings = settings_for(tmp_path, db_path=public / "private" / "library.sqlite3")
    await VectorStore(settings).initialize()
    assert stat.S_IMODE(public.stat().st_mode) == 0o755
    assert stat.S_IMODE(settings.db_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(settings.db_path.stat().st_mode) == 0o600
    with pytest.raises(StoreError, match="private directory"):
        await VectorStore(settings_for(tmp_path, db_path=public / "unsafe.sqlite3")).initialize()
    assert stat.S_IMODE(public.stat().st_mode) == 0o755


async def test_symlink_and_hardlink_paths_do_not_mutate_targets(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"do not alter")
    alias = tmp_path / "alias.sqlite3"
    alias.symlink_to(target)
    with pytest.raises(StoreError, match="symbolic links"):
        await VectorStore(settings_for(tmp_path, db_path=alias)).initialize()
    alias.unlink()
    os.link(target, alias)
    with pytest.raises(StoreError, match="hard links"):
        await VectorStore(settings_for(tmp_path, db_path=alias)).initialize()
    assert target.read_bytes() == b"do not alter"


async def test_path_substitution_after_initialization_is_rejected(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    await store.initialize()
    target = tmp_path / "unrelated.sqlite3"
    target.write_bytes(b"do not alter")
    settings.db_path.unlink()
    settings.db_path.symlink_to(target)
    with pytest.raises(StoreError, match="symbolic links"):
        await store.list_documents()
    assert target.read_bytes() == b"do not alter"


async def test_named_but_altered_index_trigger_is_preserved_and_rejected(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    await VectorStore(settings).initialize()
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("DROP TRIGGER chunks_ai")
        conn.execute("CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN SELECT 1; END")
    before = settings.db_path.read_bytes()
    with pytest.raises(StoreError, match="schema objects"):
        await VectorStore(settings).initialize()
    assert settings.db_path.read_bytes() == before


async def test_cancelled_pending_write_preserves_previous_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    old, chunks = source("cancel", ["oldword"])
    await store.upsert_document(old, chunks, [[1, 0, 0]])
    new, replacement = source("cancel", ["newword"], "two")
    entered = threading.Event()
    original = store._upsert

    def observed(*args: object) -> bool:
        entered.set()
        return original(*args)

    monkeypatch.setattr(store, "_upsert", observed)
    with sqlite3.connect(settings.db_path, isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        pending = asyncio.create_task(store.upsert_document(new, replacement, [[1, 0, 0]]))
        assert await asyncio.to_thread(entered.wait, 5)
        pending.cancel()
        await asyncio.sleep(0)
        blocker.execute("COMMIT")
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert await store.get_document(old.id) == old


async def test_bounded_lock_failure_is_actionable_and_preserves_data(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, sqlite_busy_timeout_ms=20)
    store = VectorStore(settings)
    old, chunks = source("busy", ["oldword"])
    await store.upsert_document(old, chunks, [[1, 0, 0]])
    new, replacement = source("busy", ["newword"], "two")
    with sqlite3.connect(settings.db_path, isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(StoreError, match="Library is busy"):
            await store.upsert_document(new, replacement, [[1, 0, 0]])
        blocker.execute("ROLLBACK")
    assert await store.get_document(old.id) == old


async def test_live_evidence_reload_tracks_replacement_and_deletion(tmp_path: Path) -> None:
    store = VectorStore(settings_for(tmp_path))
    document, chunks = source("reload", ["first evidence", "second evidence"])
    await store.upsert_document(document, chunks, [[1, 0, 0], [0, 1, 0]])
    hits = await store.get_hits_by_ids([chunks[1].id, chunks[0].id])
    assert [hit.chunk for hit in hits] == [chunks[1], chunks[0]]
    assert all(hit.source_uri == document.source_uri for hit in hits)
    assert await store.get_hits_by_ids(["missing"]) == []
    replacement, newer = source("reload", ["changed evidence"], "new")
    await store.upsert_document(replacement, newer, [[1, 0, 0]])
    assert await store.get_hits_by_ids([chunks[0].id]) == []
    assert (await store.get_hits_by_ids([newer[0].id]))[0].chunk == newer[0]
    await store.delete_document(document.id)
    assert await store.get_hits_by_ids([newer[0].id]) == []


@pytest.mark.parametrize(
    "ids", [["repeated", "repeated"], [""], ["x" * 129], [str(i) for i in range(51)]]
)
async def test_live_evidence_reload_rejects_unbounded_or_ambiguous_ids(
    tmp_path: Path, ids: list[str]
) -> None:
    store = VectorStore(settings_for(tmp_path))
    with pytest.raises(StoreError):
        await store.get_hits_by_ids(ids)
