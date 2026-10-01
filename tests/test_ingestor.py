from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter, get_configuration
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from mimir_rag.config import Settings
from mimir_rag.errors import DocumentError, ProviderError
from mimir_rag.ingestor import Ingestor, _prepare
from mimir_rag.models import Chunk, Document
from mimir_rag.tokenization import get_encoding


class MemoryStore:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.writes = 0

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def document_chunk_count(self, document_id: str) -> int:
        return len(self.chunks.get(document_id, []))

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> bool:
        assert len(chunks) == len(embeddings)
        previous = self.documents.get(document.id)
        if previous is not None and previous.generation == document.generation:
            return False
        self.documents[document.id] = document
        self.chunks[document.id] = chunks
        self.writes += 1
        return True


class Embeddings:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_at = fail_at

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(self.calls) == self.fail_at:
            raise ProviderError("Embedding timeout after bounded retries.")
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "library.sqlite3",
        embedding_dimensions=3,
        chunk_target_tokens=32,
        chunk_max_tokens=48,
        chunk_overlap_tokens=8,
        embedding_batch_size=2,
    )


def write_pdf(
    path: Path,
    contents: list[bytes | None],
    *,
    encrypted: bool = False,
    compressed: bool = False,
) -> None:
    writer = PdfWriter()
    for content in contents:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        if content is not None:
            stream = StreamObject()
            stream.set_data(content)
            page[NameObject("/Contents")] = writer._add_object(stream)
            if compressed:
                page.compress_content_streams()
    writer.add_metadata({"/Title": "Synthetic Handbook", "/Author": "Synthetic Author"})
    if encrypted:
        writer.encrypt("synthetic-password")
    with path.open("wb") as output:
        writer.write(output)


def test_markdown_metadata_and_exact_unicode_provenance(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "knowledge.md"
    text = (
        "---\r\ntitle: Source Book\r\nauthor: Source Author\r\n---\r\n"
        "# Feedback\r\n\r\n**feedback loop** is a core theory. 먼저 관찰해야 합니다. 🧠\r\n"
        "\r\n## Practice\r\nYou should follow the model steps.\r\n"
    )
    path.write_bytes(text.encode())
    document, chunks = _prepare(path, settings, None, None, "licensed")
    assert document.title == "Source Book"
    assert document.author == "Source Author"
    assert document.content_hash == hashlib.sha256(text.encode()).hexdigest()
    assert "Markdown front matter" in document.metadata_origin
    assert "heuristic" in document.metadata_origin
    assert chunks
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.line_start == text[: chunk.char_start].count("\n") + 1
        assert chunk.line_end == text[: chunk.char_end - 1].count("\n") + 1
        assert chunk.page is None
        assert chunk.tokens <= settings.chunk_max_tokens
        assert chunk.tags
        assert "title: Source Book" not in chunk.text
        assert not ("# Feedback" in chunk.text and "## Practice" in chunk.text)
    assert any("feedback loop" in chunk.concepts for chunk in chunks)
    assert any("Actionable Advice" in chunk.tags for chunk in chunks)


def test_user_overrides_parser_metadata(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "test.md"
    path.write_text("---\ntitle: Parser Title\nauthor: Parser Author\n---\n# Heading\nText.")
    document, chunks = _prepare(path, settings, "Explicit Title", "Explicit Author", "permission")
    assert document.title == "Explicit Title"
    assert document.author == "Explicit Author"
    assert all(
        chunk.title == document.title and chunk.author == document.author for chunk in chunks
    )
    assert document.metadata_origin.count("user override") == 2


def test_unknown_author_and_uri_filename_decoding(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "한 글 #book.txt"
    path.write_text("A factual paragraph.")
    document, _ = _prepare(path, settings, None, None, "authorized")
    assert document.title == "한 글 #book"
    assert document.author is None
    assert document.source_uri == path.resolve().as_uri()


def test_heading_hierarchy_and_fenced_code(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "headings.md"
    path.write_text(
        "# Book\n\n## Theory\nExplanation.\n\n```md\n# Fake title\n```\n\n## Practice\nAdvice."
    )
    document, chunks = _prepare(path, settings, None, None, "authorized")
    assert document.title == "Book"
    assert {chunk.section for chunk in chunks} == {"Book", "Book / Theory", "Book / Practice"}
    assert all(chunk.section != "Fake title" for chunk in chunks)


def test_oversized_multilingual_sentence_splits_without_unicode_damage(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "unicode.txt"
    text = "뇌🧠漢字🌌é" * 120
    path.write_text(text)
    _, chunks = _prepare(path, settings, None, None, "authorized")
    encoding = get_encoding()
    assert len(chunks) > 1
    covered: set[int] = set()
    for index, chunk in enumerate(chunks):
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert "\ufffd" not in chunk.text
        assert chunk.tokens == len(encoding.encode(chunk.text)) <= settings.chunk_max_tokens
        covered.update(range(chunk.char_start, chunk.char_end))
        if index:
            previous = chunks[index - 1]
            overlap = text[chunk.char_start : previous.char_end]
            assert len(encoding.encode(overlap)) <= settings.chunk_overlap_tokens
    assert covered == set(range(len(text)))


def test_hash_and_parse_share_one_snapshot(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mimir_rag import ingestor

    path = tmp_path / "snapshot.txt"
    original = "Original source theory."
    path.write_text(original)
    sections = ingestor._sections

    def rewrite_after_snapshot(text: str, start: int, page: int | None, markdown: bool):
        path.write_text("Changed on disk after read.")
        return sections(text, start, page, markdown)

    monkeypatch.setattr(ingestor, "_sections", rewrite_after_snapshot)
    document, chunks = _prepare(path, settings, None, None, "authorized")
    assert document.content_hash == hashlib.sha256(original.encode()).hexdigest()
    assert chunks[0].text == original


@pytest.mark.asyncio
async def test_identical_ingestion_avoids_api_and_metadata_change_replaces(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "book.txt"
    path.write_text("A feedback loop is a core theory.")
    store, provider = MemoryStore(), Embeddings()
    engine = Ingestor(settings, store, provider)
    first = await engine.ingest(path)
    chunk_ids = [chunk.id for chunk in store.chunks[first.document_id]]
    calls = len(provider.calls)
    same = await engine.ingest(path)
    assert same.status == "unchanged"
    assert len(provider.calls) == calls
    assert store.writes == 1
    changed = await engine.ingest(path, title="New title")
    assert changed.document_id == first.document_id
    assert changed.status == "indexed"
    assert [chunk.id for chunk in store.chunks[first.document_id]] != chunk_ids
    assert store.writes == 2


@pytest.mark.asyncio
async def test_content_and_chunk_configuration_change_generation(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "book.txt"
    path.write_text("The first source theory.")
    store, provider = MemoryStore(), Embeddings()
    first = await Ingestor(settings, store, provider).ingest(path)
    initial_generation = store.documents[first.document_id].generation
    path.write_text("The second source theory.")
    changed = await Ingestor(settings, store, provider).ingest(path)
    content_generation = store.documents[first.document_id].generation
    assert changed.document_id == first.document_id
    assert content_generation != initial_generation
    changed_settings = settings.model_copy(update={"chunk_target_tokens": 40})
    await Ingestor(changed_settings, store, provider).ingest(path)
    assert store.documents[first.document_id].generation != content_generation


@pytest.mark.asyncio
async def test_mid_batch_timeout_preserves_previous_generation(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("A source theory.")
    store = MemoryStore()
    result = await Ingestor(settings, store, Embeddings()).ingest(path)
    previous, chunks = store.documents[result.document_id], store.chunks[result.document_id]
    path.write_text("\n\n".join(f"Paragraph {number}: " + "knowledge " * 30 for number in range(8)))
    provider = Embeddings(fail_at=2)
    with pytest.raises(ProviderError, match="timeout"):
        await Ingestor(settings, store, provider).ingest(path)
    assert len(provider.calls) == 2
    assert store.writes == 1
    assert store.documents[result.document_id] == previous
    assert store.chunks[result.document_id] == chunks


@pytest.mark.asyncio
async def test_embedding_cancellation_does_not_publish(tmp_path: Path, settings: Settings) -> None:
    class CancelledEmbeddings(Embeddings):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise asyncio.CancelledError

    path = tmp_path / "source.txt"
    path.write_text("A source theory.")
    store = MemoryStore()
    with pytest.raises(asyncio.CancelledError):
        await Ingestor(settings, store, CancelledEmbeddings()).ingest(path)
    assert store.writes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [float("nan"), 0.0, 1.0], [1.0, 0.0]])
async def test_invalid_embeddings_never_publish(
    tmp_path: Path, settings: Settings, vector: list[float]
) -> None:
    class InvalidEmbeddings(Embeddings):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [vector for _ in texts]

    path = tmp_path / "source.txt"
    path.write_text("Source theory.")
    store = MemoryStore()
    with pytest.raises(ProviderError, match="invalid"):
        await Ingestor(settings, store, InvalidEmbeddings()).ingest(path)
    assert store.writes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("unsupported.docx", b"content", "Unsupported"),
        ("empty.txt", b"", "empty"),
        ("whitespace.txt", b" \n\t", "no extractable"),
        ("invalid.txt", b"\xff", "UTF-8"),
        ("binary.txt", b"abc\x00", "NUL"),
        ("bad.pdf", b"not a pdf", "Cannot parse PDF"),
    ],
)
async def test_invalid_documents_fail_before_api(
    tmp_path: Path, settings: Settings, name: str, contents: bytes, message: str
) -> None:
    path = tmp_path / name
    path.write_bytes(contents)
    store, provider = MemoryStore(), Embeddings()
    with pytest.raises(DocumentError, match=message):
        await Ingestor(settings, store, provider).ingest(path)
    assert provider.calls == []
    assert store.writes == 0


@pytest.mark.parametrize(
    "frontmatter",
    [
        "---\ntitle: [invalid, array]\n---\nText",
        "---\ntitle: !!python/object/apply:os.system ['echo bad']\n---\nText",
        "---\ntitle: never closed",
        "---\n- item\n---\nText",
        "---\n" + "a" * 17_000 + "\n---\nText",
    ],
)
def test_frontmatter_rejects_invalid_or_unsafe_metadata(
    tmp_path: Path, settings: Settings, frontmatter: str
) -> None:
    path = tmp_path / "bad.md"
    path.write_text(frontmatter)
    with pytest.raises(DocumentError):
        _prepare(path, settings, None, None, "authorized")


@pytest.mark.parametrize("field", ["title", "author", "rights"])
def test_metadata_limits_are_actionable(tmp_path: Path, settings: Settings, field: str) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Source theory.")
    values = {"title": None, "author": None, "rights": "authorized"}
    values[field] = "x" * (2001 if field == "rights" else 501)
    with pytest.raises(DocumentError, match="metadata limit"):
        _prepare(path, settings, values["title"], values["author"], values["rights"])


def test_resource_limits_reject_before_embedding(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Source " * 100)
    for update in ({"max_file_bytes": 20}, {"max_extracted_chars": 20}, {"max_chunks": 1}):
        with pytest.raises(DocumentError, match="limit"):
            _prepare(path, settings.model_copy(update=update), None, None, "authorized")


def test_pdf_metadata_physical_pages_and_exact_page_offsets(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "synthetic.pdf"
    write_pdf(
        path,
        [
            None,
            b"BT /F1 12 Tf (Chapter 2) Tj 0 -20 Td (A feedback loop is a core theory.) Tj ET",
            b"BT /F1 12 Tf (Third page advice.) Tj ET",
        ],
    )
    document, chunks = _prepare(path, settings, None, None, "authorized")
    reader = PdfReader(path)
    assert document.title == "Synthetic Handbook"
    assert document.author == "Synthetic Author"
    assert {chunk.page for chunk in chunks} == {2, 3}
    for chunk in chunks:
        text = reader.pages[chunk.page - 1].extract_text().replace("\r\n", "\n").replace("\r", "\n")
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.line_start is None and chunk.line_end is None


def test_pdf_encryption_scanned_pages_and_limits(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "synthetic.pdf"
    write_pdf(path, [b"BT /F1 12 Tf (Source.) Tj ET"], encrypted=True)
    with pytest.raises(DocumentError, match="Encrypted"):
        _prepare(path, settings, None, None, "authorized")
    write_pdf(path, [b"0 0 20 20 re S"])
    with pytest.raises(DocumentError, match="OCR"):
        _prepare(path, settings, None, None, "authorized")
    write_pdf(path, [None, b"BT /F1 12 Tf (Source.) Tj ET"])
    with pytest.raises(DocumentError, match="page limit"):
        _prepare(path, settings.model_copy(update={"max_pdf_pages": 1}), None, None, "authorized")


def test_pdf_decompression_is_bounded_and_configuration_restored(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "compressed.pdf"
    write_pdf(path, [b"BT /F1 12 Tf (" + b"a" * 20_000 + b") Tj ET"], compressed=True)
    before = get_configuration()
    with pytest.raises(DocumentError, match="parser limits|byte limit"):
        _prepare(
            path,
            settings.model_copy(update={"max_pdf_stream_bytes": 1000}),
            None,
            None,
            "authorized",
        )
    assert get_configuration() == before


@pytest.mark.asyncio
async def test_concurrent_identical_ingestion_has_one_document(
    tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("A source theory.")
    store, provider = MemoryStore(), Embeddings()
    engine = Ingestor(settings, store, provider)
    results = await asyncio.gather(engine.ingest(path), engine.ingest(path))
    assert results[0].document_id == results[1].document_id
    assert len(store.documents) == store.writes == 1


@pytest.mark.asyncio
async def test_sqlite_ingestion_concurrency_and_timeout_preserve_complete_rows(
    tmp_path: Path, settings: Settings
) -> None:
    from mimir_rag.vector_store import VectorStore

    path = tmp_path / "source.txt"
    path.write_text("A feedback loop is a source theory.")
    store = VectorStore(settings)
    await store.initialize()
    engine = Ingestor(settings, store, Embeddings())
    first, same = await asyncio.gather(engine.ingest(path), engine.ingest(path))
    assert first.document_id == same.document_id
    assert {first.status, same.status} == {"indexed", "unchanged"}
    previous = await store.get_document(first.document_id)
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
        counts = [
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("chunks", "chunks_fts", "concept_mentions")
        ]
    assert counts[0] == counts[1] == first.chunk_count
    assert counts[2] > 0
    path.write_text("\n\n".join(f"Paragraph {number}: " + "knowledge " * 30 for number in range(8)))
    with pytest.raises(ProviderError, match="timeout"):
        await Ingestor(settings, store, Embeddings(fail_at=2)).ingest(path)
    assert await store.get_document(first.document_id) == previous
    with sqlite3.connect(settings.db_path) as connection:
        after = [
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("chunks", "chunks_fts", "concept_mentions")
        ]
    assert after == counts


def test_nonregular_source_is_rejected_without_blocking(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "pipe.txt"
    os.mkfifo(path)
    with pytest.raises(DocumentError, match="regular"):
        _prepare(path, settings, None, None, "authorized")


@pytest.mark.asyncio
async def test_embedding_count_mismatch_never_publishes(tmp_path: Path, settings: Settings) -> None:
    class MissingEmbeddings(Embeddings):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    path = tmp_path / "source.txt"
    path.write_text("A source theory.")
    store = MemoryStore()
    with pytest.raises(ProviderError, match="count"):
        await Ingestor(settings, store, MissingEmbeddings()).ingest(path)
    assert store.writes == 0
