from pathlib import Path

import pytest

from mimir_rag.config import Settings
from mimir_rag.errors import StoreError
from mimir_rag.models import Chunk, Document
from mimir_rag.vector_store import VectorStore


async def test_large_library_search_and_capacity_rollback(tmp_path: Path) -> None:
    capacity = 50_001
    settings = Settings(
        db_path=tmp_path / "private" / "library.sqlite3",
        max_chunks=capacity,
        embedding_dimensions=3,
        dense_batch_size=512,
    )
    store = VectorStore(settings)
    document = Document(
        id="large-fixture",
        source_uri="file:///synthetic/large.md",
        content_hash="synthetic-large-content",
        generation="complete",
        title="Synthetic capacity fixture",
        author=None,
        rights="Synthetic test data",
        metadata_origin="test fixture",
        ingest_signature=settings.ingest_signature,
    )
    chunks = []
    for ordinal in range(capacity):
        text = "tailneedle" if ordinal == capacity - 1 else "ordinary"
        chunks.append(
            Chunk(
                id=f"fixture-{ordinal:06d}",
                document_id=document.id,
                ordinal=ordinal,
                text=text,
                title=document.title,
                section="Synthetic section",
                char_start=0,
                char_end=len(text),
                tokens=1,
                tags=[],
                concepts=[],
            )
        )
    vectors = [[1, 0, 0]] * (capacity - 1) + [[0, 1, 0]]
    assert await store.upsert_document(document, chunks, vectors)
    hits = await store.search("tailneedle", [0, 1, 0], top_k=1)
    assert hits[0].chunk.id == chunks[-1].id
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_score == pytest.approx(1)
    assert hits[0].source_uri == document.source_uri

    overflow = document.model_copy(update={"id": "overflow", "generation": "overflow"})
    overflow_chunk = chunks[0].model_copy(
        update={"id": "overflow-0", "document_id": overflow.id, "ordinal": 0}
    )
    with pytest.raises(StoreError, match="capacity"):
        await store.upsert_document(overflow, [overflow_chunk], [[1, 0, 0]])
    assert await store.get_document(overflow.id) is None
    assert await store.get_document(document.id) == document
    assert (await store.search("tailneedle", [0, 1, 0], top_k=1))[0].chunk == chunks[-1]
