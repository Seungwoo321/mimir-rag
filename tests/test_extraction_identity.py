from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from fontTools.fontBuilder import FontBuilder  # type: ignore[import-untyped]
from fontTools.pens.t2CharStringPen import T2CharStringPen  # type: ignore[import-untyped]
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
)

from mimir_rag import ingestor
from mimir_rag.config import Settings
from mimir_rag.ingestor import Ingestor, _prepare
from mimir_rag.providers import ProviderClient
from mimir_rag.vector_store import VectorStore


def custom_cff_pdf(path: Path) -> None:
    glyphs = [".notdef", "alpha"]
    font = FontBuilder(1000, isTTF=False)
    font.setupGlyphOrder(glyphs)
    font.setupCharacterMap({0x03B1: "alpha"})
    font.setupHorizontalMetrics({glyph: (600, 0) for glyph in glyphs})
    font.setupHorizontalHeader(ascent=800, descent=-200)
    font.setupNameTable({"familyName": "Synthetic", "styleName": "Regular"})
    font.setupOS2()
    font.setupPost()
    strings = {glyph: T2CharStringPen(600, None).getCharString() for glyph in glyphs}
    font.setupCFF("SyntheticCFF", {"FullName": "Synthetic CFF"}, strings, {})
    cff = font.font["CFF "].cff
    encoding = [".notdef"] * 256
    encoding[65] = "alpha"
    cff.topDictIndex[0].Encoding = encoding
    font_bytes = io.BytesIO()
    cff.compile(font_bytes, font.font)

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    embedded = StreamObject()
    embedded.set_data(font_bytes.getvalue())
    embedded[NameObject("/Subtype")] = NameObject("/Type1C")
    descriptor = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject("/SyntheticCFF"),
            NameObject("/FontFile3"): writer._add_object(embedded),
        }
    )
    resource = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/SyntheticCFF"),
            NameObject("/FontDescriptor"): writer._add_object(descriptor),
            NameObject("/FirstChar"): NumberObject(65),
            NameObject("/LastChar"): NumberObject(65),
            NameObject("/Widths"): ArrayObject([NumberObject(600)]),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(resource)})}
    )
    content = StreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (A) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(path)


class Embeddings(ProviderClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_embedded_cff_custom_encoding_preserves_unicode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "font.pdf"
    custom_cff_pdf(path)
    _, chunks = _prepare(path, Settings(), "Synthetic font", None, "synthetic")
    assert [chunk.text.strip() for chunk in chunks] == ["α"]
    assert chunks[0].page == 1
    assert "fontTools is required" not in caplog.text


async def test_decoder_upgrade_reindexes_same_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pypdf._font as pdf_font

    path = tmp_path / "font.pdf"
    custom_cff_pdf(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=3)
    store = VectorStore(settings)
    async with Embeddings(settings) as provider:
        engine = Ingestor(settings, store, provider)
        monkeypatch.setattr(pdf_font, "HAS_FONTTOOLS", False)
        first = await engine.ingest(path)
        original = await store.get_document(first.document_id)
        assert original is not None and original.content_hash == source_hash
        assert provider.calls == [["A"]]
        monkeypatch.setattr(pdf_font, "HAS_FONTTOOLS", True)
        changed = await engine.ingest(path)
        current = await store.get_document(first.document_id)
        assert current is not None
        assert current.content_hash == source_hash
        assert current.generation != original.generation
        assert changed.status == "indexed"
        assert provider.calls == [["A"], ["α"]]
        hits = await store.search("α", [1, 0, 0])
        assert hits[0].chunk.text == "α" and hits[0].chunk.page == 1
        repeated = await engine.ingest(path)
        assert repeated.status == "unchanged"
        assert len(provider.calls) == 2


@pytest.mark.parametrize("field", ["page", "section", "offset"])
async def test_changed_extraction_citation_coordinates_refresh_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Stable source text.")
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=3)
    parsed = ingestor._parse_source(path, settings)
    unit = parsed.units[0]
    store = VectorStore(settings)
    async with Embeddings(settings) as provider:
        engine = Ingestor(settings, store, provider)
        first = await engine.ingest(path)
        original = await store.get_document(first.document_id)
        assert original is not None
        if field == "page":
            changed_unit = replace(unit, page=2)
        elif field == "section":
            changed_unit = replace(unit, section="Corrected chapter")
        else:
            changed_unit = replace(unit, text="\n" + unit.text, start=1, end=unit.end + 1)
        changed_parse = replace(parsed, units=(changed_unit,))
        monkeypatch.setattr(ingestor, "_parse_source", lambda *args: changed_parse)
        result = await engine.ingest(path)
        current = await store.get_document(first.document_id)
        assert current is not None
        assert current.content_hash == original.content_hash
        assert current.generation != original.generation
        assert result.status == "indexed"
        assert provider.calls == [[unit.text], [unit.text]]


async def test_excluded_prefix_line_change_refreshes_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Stable source bytes.")
    settings = Settings(db_path=tmp_path / "library.sqlite3", embedding_dimensions=3)
    parsed = ingestor._parse_source(path, settings)
    text = "Stable indexed text."
    unit = replace(parsed.units[0], text="\n\n" + text, start=2, end=2 + len(text))
    current_parse = replace(parsed, units=(unit,))
    monkeypatch.setattr(ingestor, "_parse_source", lambda *args: current_parse)
    store = VectorStore(settings)
    async with Embeddings(settings) as provider:
        engine = Ingestor(settings, store, provider)
        first = await engine.ingest(path)
        original = await store.get_document(first.document_id)
        assert original is not None
        initial = await store.search(text, [1, 0, 0])
        assert initial[0].chunk.line_start == initial[0].chunk.line_end == 3
        current_parse = replace(parsed, units=(replace(unit, text="xx" + text),))
        changed = await engine.ingest(path)
        current = await store.get_document(first.document_id)
        assert current is not None
        assert current.content_hash == original.content_hash
        assert current.generation != original.generation
        assert changed.status == "indexed"
        hits = await store.search(text, [1, 0, 0])
        assert hits[0].chunk.text == text
        assert hits[0].chunk.char_start == 2
        assert hits[0].chunk.line_start == hits[0].chunk.line_end == 1
        repeated = await engine.ingest(path)
        assert repeated.status == "unchanged"
        assert provider.calls == [[text], [text]]
