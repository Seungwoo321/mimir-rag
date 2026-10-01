from __future__ import annotations

from pathlib import Path

from mimir_rag.config import Settings
from mimir_rag.ingestor import Ingestor, _prepare


async def test_unchanged_result_stays_bound_to_its_generation(tmp_path: Path):
    source = tmp_path / "source.md"
    source.write_text("# Example\n\nA feedback loop is a core theory.")
    settings = Settings(db_path=tmp_path / "private" / "library.sqlite3")
    document, chunks = _prepare(source, settings, None, None, "authorized")

    class ConcurrentStore:
        async def get_document(self, document_id):
            assert document_id == document.id
            return document

        async def document_chunk_count(self, document_id):
            raise AssertionError("A second snapshot could describe a concurrent new generation.")

    class UnusedProvider:
        async def embed(self, texts):
            raise AssertionError("Identical ingestion must not call the provider.")

    result = await Ingestor(settings, ConcurrentStore(), UnusedProvider()).ingest(
        source, rights="authorized"
    )
    assert result.status == "unchanged"
    assert result.document_id == document.id
    assert result.chunk_count == len(chunks)
