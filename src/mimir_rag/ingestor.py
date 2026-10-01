from __future__ import annotations

import asyncio
import bisect
import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import yaml
from pypdf import PdfReader, apply_configuration
from pypdf.errors import PyPdfError
from pypdf.generic import DictionaryObject, StreamObject

from .errors import DocumentError, ProviderError
from .models import Chunk, Document, IngestResult
from .tokenization import get_encoding

if TYPE_CHECKING:
    from .config import Settings
    from .providers import ProviderClient
    from .vector_store import VectorStore


_ENCODING = get_encoding()
_FRONTMATTER_LIMIT = 16_384
_ATX = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_CHAPTER = re.compile(r"^(?:chapter|part|section|장|제\s*\d+\s*장)\b[^\n]{0,990}$", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+")
_TAG_RULES = (
    (
        "Framework",
        re.compile(r"\b(framework|model|method|steps|process)\b|프레임워크|방법론", re.I),
    ),
    (
        "Actionable Advice",
        re.compile(r"\b(should|must|recommend|avoid|how to|step)\b|해야|권장|먼저", re.I),
    ),
    (
        "Core Theory",
        re.compile(r"\b(theory|principle|definition|because|concept)\b|이론|원리|개념", re.I),
    ),
)
_CONCEPT_PATTERNS = (
    re.compile(r"\*\*([^*\n]{2,80})\*\*"),
    re.compile(r'["“]([^"”\n]{2,80})["”]'),
    re.compile(r"\b([A-Z][\w-]*(?: [A-Z][\w-]*){1,4})\b"),
    re.compile(
        r"\b([\w-]+(?: [\w-]+){0,2} (?:theory|principle|framework|model|loop|method))\b", re.I
    ),
)
_VISIBLE_OPERATORS = frozenset(
    (
        b"Do",
        b"BI",
        b"Tj",
        b"TJ",
        b"'",
        b'"',
        b"S",
        b"s",
        b"f",
        b"F",
        b"f*",
        b"B",
        b"B*",
        b"b",
        b"b*",
        b"sh",
    )
)


@dataclass(frozen=True)
class _Unit:
    text: str
    start: int
    end: int
    section: str
    page: int | None


@dataclass(frozen=True)
class _ParsedSource:
    source_uri: str
    content_hash: str
    title: str | None
    author: str | None
    title_origin: str
    author_origin: str
    units: tuple[_Unit, ...]


def _tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def _metadata(value: object, name: str, maximum: int, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise DocumentError(f"{name.capitalize()} cannot be empty.")
        return None
    if not isinstance(value, str):
        raise DocumentError(f"{name.capitalize()} metadata must be a string.")
    result = value.strip()
    if not result:
        if required:
            raise DocumentError(f"{name.capitalize()} cannot be empty.")
        return None
    if len(result) > maximum:
        raise DocumentError(f"{name.capitalize()} exceeds the {maximum}-character metadata limit.")
    if "\x00" in result:
        raise DocumentError(f"{name.capitalize()} contains an invalid NUL character.")
    return result


def _read_snapshot(path: Path, maximum: int) -> tuple[Path, bytes]:
    try:
        canonical = path.expanduser().resolve(strict=True)
        descriptor = os.open(canonical, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            attributes = os.fstat(stream.fileno())
            if not stat.S_ISREG(attributes.st_mode):
                raise DocumentError("Source must be a regular PDF, TXT, or Markdown file.")
            if attributes.st_size > maximum:
                raise DocumentError(f"Source exceeds the configured {maximum}-byte file limit.")
            data = stream.read(maximum + 1)
    except (OSError, RuntimeError) as exc:
        raise DocumentError(
            "Cannot read source file. Check its path and read permissions."
        ) from exc
    if len(data) > maximum:
        raise DocumentError(f"Source exceeds the configured {maximum}-byte file limit.")
    if not data:
        raise DocumentError("Source file is empty; select a document containing text.")
    return canonical, data


def _frontmatter(text: str) -> tuple[dict[str, object], int]:
    initial = 1 if text.startswith("\ufeff") else 0
    first_line = re.match(r"---[ \t]*(?:\r?\n)", text[initial:])
    if first_line is None:
        return {}, initial
    start = initial + first_line.end()
    closing = re.search(r"(?m)^---[ \t]*\r?$", text[start : start + _FRONTMATTER_LIMIT + 1])
    if closing is None:
        raise DocumentError("Markdown front matter is unclosed or exceeds 16,384 characters.")
    end = start + closing.start()
    if end - start > _FRONTMATTER_LIMIT:
        raise DocumentError("Markdown front matter exceeds 16,384 characters.")
    try:
        values = yaml.safe_load(text[start:end])
    except (yaml.YAMLError, RecursionError) as exc:
        raise DocumentError(
            "Invalid Markdown YAML front matter; use a mapping of metadata fields."
        ) from exc
    if values is None:
        values = {}
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise DocumentError("Markdown YAML front matter must be a mapping with string keys.")
    body_start = start + closing.end()
    if text[body_start : body_start + 1] == "\n":
        body_start += 1
    return values, body_start


def _sections(
    text: str, start: int, page: int | None, markdown: bool
) -> tuple[list[_Unit], str | None]:
    lines = list(re.finditer(r"[^\n]*(?:\n|$)", text[start:]))
    boundaries: list[tuple[int, str]] = [(start, "Document" if page is None else f"Page {page}")]
    headings: dict[int, str] = {}
    title: str | None = None
    fence: str | None = None
    for index, line_match in enumerate(lines):
        line = line_match.group().rstrip("\r\n")
        position = start + line_match.start()
        fenced = re.match(r"^ {0,3}(`{3,}|~{3,})", line) if markdown else None
        if fenced:
            delimiter = fenced.group(1)
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        match = _ATX.match(line) if markdown else None
        level = len(match.group(1)) if match else 1
        label = match.group(2).strip() if match else None
        if label is None and markdown and index + 1 < len(lines) and line.strip():
            underline = lines[index + 1].group().strip()
            if re.fullmatch(r"={3,}|-{3,}", underline):
                label, level = line.strip(), 1 if underline[0] == "=" else 2
        if label is None and _CHAPTER.match(line.strip()):
            label = line.strip()
        if label is None:
            continue
        if title is None and markdown and level == 1:
            title = label
        headings = {key: value for key, value in headings.items() if key < level}
        headings[level] = label
        section = " / ".join(headings[key] for key in sorted(headings))[:1000]
        if boundaries[-1][0] == position:
            boundaries[-1] = (position, section)
        else:
            boundaries.append((position, section))
    units = [
        _Unit(
            text,
            boundary,
            boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text),
            section,
            page,
        )
        for index, (boundary, section) in enumerate(boundaries)
    ]
    return units, title


def _bounded_forms(page: DictionaryObject, maximum: int, content_size: int) -> None:
    resources = page.get("/Resources")
    pending = [resources.get_object()] if resources is not None else []
    visited: set[int] = set()
    total = content_size
    while pending:
        resource = pending.pop()
        if not isinstance(resource, DictionaryObject):
            raise DocumentError("PDF contains an invalid page resource dictionary.")
        objects = resource.get("/XObject")
        if objects is None:
            continue
        object_dictionary = objects.get_object()
        if not isinstance(object_dictionary, DictionaryObject):
            raise DocumentError("PDF contains an invalid external-object dictionary.")
        for reference in object_dictionary.values():
            obj = reference.get_object()
            if not isinstance(obj, StreamObject) or obj.get("/Subtype") != "/Form":
                continue
            identity = id(obj)
            if identity in visited:
                continue
            visited.add(identity)
            if len(visited) > 1000:
                raise DocumentError("PDF page exceeds the 1,000-form resource limit.")
            total += len(obj.get_data())
            if total > maximum:
                raise DocumentError(
                    "PDF page content and form streams exceed the configured byte limit."
                )
            nested = obj.get("/Resources")
            if nested is not None:
                pending.append(nested.get_object())


def _parse_pdf(data: bytes, settings: Settings) -> tuple[list[_Unit], str | None, str | None]:
    maximum = settings.max_pdf_stream_bytes
    units: list[_Unit] = []
    extracted = 0
    with apply_configuration(
        maximum_declared_stream_length=maximum,
        array_based_stream_maximum_output_length=maximum,
        zlib_maximum_output_length=maximum,
        lzw_maximum_output_length=maximum,
        run_length_maximum_output_length=maximum,
        jbig2_maximum_output_length=maximum,
        zlib_maximum_recovery_input_length=min(maximum, 100_000),
        page_tree_maximum_entries=settings.max_pdf_pages * 4 + 32,
        page_tree_maximum_depth=64,
        xform_maximum_invocations_per_extraction=1000,
    ):
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise DocumentError(
                    "Encrypted PDFs are unsupported; export an authorized unencrypted copy."
                )
            if len(reader.pages) > settings.max_pdf_pages:
                raise DocumentError(
                    f"PDF exceeds the configured {settings.max_pdf_pages}-page limit."
                )
            metadata = reader.metadata
            title = _metadata(metadata.title if metadata else None, "title", 500)
            author = _metadata(metadata.author if metadata else None, "author", 500)
            for number, page in enumerate(reader.pages, 1):
                contents = page.get_contents()
                content_size = len(contents.get_data()) if contents is not None else 0
                if content_size > maximum:
                    raise DocumentError(
                        "PDF page content stream exceeds the configured byte limit."
                    )
                _bounded_forms(page, maximum, content_size)
                text = page.extract_text().replace("\r\n", "\n").replace("\r", "\n")
                extracted += len(text)
                if extracted > settings.max_extracted_chars:
                    raise DocumentError(
                        "PDF extracted text exceeds the configured character limit."
                    )
                if not text.strip():
                    if contents is not None and any(
                        operator in _VISIBLE_OPERATORS for _, operator in contents.operations
                    ):
                        raise DocumentError(
                            f"PDF page {number} has no extractable text; preprocess it with OCR."
                        )
                    continue
                page_units, _ = _sections(text, 0, number, False)
                units.extend(page_units)
        except DocumentError:
            raise
        except (PyPdfError, OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise DocumentError(
                "Cannot parse PDF safely; it is corrupt or exceeds configured parser limits."
            ) from exc
    return units, title, author


def _parse_source(path: Path, settings: Settings) -> _ParsedSource:
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
        raise DocumentError("Unsupported source format. Select a PDF, UTF-8 TXT, or Markdown file.")
    canonical, data = _read_snapshot(path, settings.max_file_bytes)
    title: str | None = None
    author: str | None = None
    title_origin = "filename"
    author_origin = "unknown"
    if suffix == ".pdf":
        units, title, author = _parse_pdf(data, settings)
        title_origin = "PDF metadata" if title else title_origin
        author_origin = "PDF metadata" if author else author_origin
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentError(
                "Text must be valid UTF-8; convert its encoding before ingestion."
            ) from exc
        if "\x00" in text:
            raise DocumentError(
                "Text contains NUL bytes; select a UTF-8 text document rather than binary data."
            )
        if len(text) > settings.max_extracted_chars:
            raise DocumentError("Source text exceeds the configured character limit.")
        markdown = suffix in {".md", ".markdown"}
        values, body_start = _frontmatter(text) if markdown else ({}, 0)
        units, heading_title = _sections(text, body_start, None, markdown)
        title = _metadata(values.get("title"), "title", 500)
        author = _metadata(values.get("author"), "author", 500)
        if title:
            title_origin = "Markdown front matter"
        elif heading_title:
            title = _metadata(heading_title, "title", 500)
            title_origin = "heading heuristic"
        if author:
            author_origin = "Markdown front matter"
    if not any(unit.text[unit.start : unit.end].strip() for unit in units):
        raise DocumentError(
            "Source contains no extractable text; image-only PDFs require OCR preprocessing."
        )
    return _ParsedSource(
        canonical.as_uri(),
        hashlib.sha256(data).hexdigest(),
        title,
        author,
        title_origin,
        author_origin,
        tuple(units),
    )


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_oversized(text: str, start: int, end: int, maximum: int) -> Iterator[tuple[int, int]]:
    while start < end:
        width = maximum
        bound = min(end, start + width)
        count = _tokens(text[start:bound])
        while bound < end and count <= maximum:
            width *= 2
            bound = min(end, start + width)
            count = _tokens(text[start:bound])
        if count <= maximum:
            yield start, end
            return
        low, high = start + 1, bound - 1
        best = start
        while low <= high:
            middle = (low + high) // 2
            if _tokens(text[start:middle]) <= maximum:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best == start:
            raise DocumentError("A source character exceeds the configured token limit.")
        yield start, best
        start = best


def _semantic_spans(unit: _Unit, target: int, maximum: int) -> Iterator[tuple[int, int]]:
    boundaries = [unit.start]
    boundaries.extend(
        unit.start + match.end()
        for match in re.finditer(r"\n[ \t\r]*\n", unit.text[unit.start : unit.end])
    )
    boundaries.append(unit.end)
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        left, right = _trim(unit.text, left, right)
        if left == right:
            continue
        if _tokens(unit.text[left:right]) <= target:
            yield left, right
            continue
        sentences = [left]
        sentences.extend(
            left + match.end() for match in _SENTENCE_BREAK.finditer(unit.text[left:right])
        )
        sentences.append(right)
        for sentence_start, sentence_end in zip(sentences, sentences[1:], strict=False):
            sentence_start, sentence_end = _trim(unit.text, sentence_start, sentence_end)
            if sentence_start < sentence_end:
                yield from _split_oversized(unit.text, sentence_start, sentence_end, maximum)


def _overlap_start(text: str, start: int, end: int, maximum: int) -> int:
    if maximum == 0:
        return end
    low, high, best = start, end, end
    while low <= high:
        middle = (low + high) // 2
        if _tokens(text[middle:end]) <= maximum:
            best, high = middle, middle - 1
        else:
            low = middle + 1
    return _trim(text, best, end)[0]


def _chunk_spans(unit: _Unit, settings: Settings) -> Iterator[tuple[int, int]]:
    current: tuple[int, int] | None = None
    for start, end in _semantic_spans(
        unit, settings.chunk_target_tokens, settings.chunk_max_tokens
    ):
        if current is None:
            current = (start, end)
            continue
        if _tokens(unit.text[current[0] : end]) <= settings.chunk_target_tokens:
            current = (current[0], end)
            continue
        yield current
        overlap = _overlap_start(unit.text, current[0], current[1], settings.chunk_overlap_tokens)
        if _tokens(unit.text[overlap:end]) > settings.chunk_max_tokens:
            overlap = start
        current = (min(start, overlap), end)
    if current is not None:
        yield current


def _concepts(text: str) -> list[str]:
    labels: list[str] = []
    for pattern in _CONCEPT_PATTERNS:
        for match in pattern.finditer(text):
            label = " ".join(match.group(1).split()).casefold()
            if 2 <= len(label) <= 80 and label not in labels:
                labels.append(label)
            if len(labels) == 12:
                return labels
    return labels


def _prepare(
    path: Path, settings: Settings, title: str | None, author: str | None, rights: str
) -> tuple[Document, list[Chunk]]:
    rights_value = _metadata(rights, "rights", 2000, required=True)
    parsed = _parse_source(path, settings)
    explicit_title = _metadata(title, "title", 500, required=title is not None)
    explicit_author = _metadata(author, "author", 500, required=author is not None)
    effective_title: str | None = (
        explicit_title or parsed.title or Path(unquote(urlsplit(parsed.source_uri).path)).stem
    )
    effective_title = _metadata(effective_title, "title", 500, required=True)
    if effective_title is None or rights_value is None:
        raise DocumentError("Required document metadata cannot be empty.")
    effective_author = explicit_author or parsed.author
    origin = (
        f"title:{'user override' if explicit_title else parsed.title_origin}; "
        f"author:{'user override' if explicit_author else parsed.author_origin}; "
        "section/tags/concepts:heuristic"
    )
    identity = hashlib.sha256(parsed.source_uri.encode()).hexdigest()
    generation_payload = {
        "content": parsed.content_hash,
        "title": effective_title,
        "author": effective_author,
        "rights": rights_value,
        "origin": origin,
        "ingest": settings.ingest_signature,
    }
    generation = hashlib.sha256(
        json.dumps(generation_payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    document = Document(
        id=identity,
        source_uri=parsed.source_uri,
        content_hash=parsed.content_hash,
        generation=generation,
        title=effective_title,
        author=effective_author,
        rights=rights_value,
        metadata_origin=origin,
        ingest_signature=settings.ingest_signature,
    )
    chunks: list[Chunk] = []
    line_breaks = (
        [match.start() for match in re.finditer("\n", parsed.units[0].text)]
        if parsed.units[0].page is None
        else []
    )
    for unit in parsed.units:
        for start, end in _chunk_spans(unit, settings):
            text = unit.text[start:end]
            ordinal = len(chunks)
            if ordinal >= settings.max_chunks:
                raise DocumentError(
                    f"Document exceeds the configured {settings.max_chunks}-chunk limit."
                )
            tags = [label for label, pattern in _TAG_RULES if pattern.search(text)] or [
                "Core Theory"
            ]
            chunks.append(
                Chunk(
                    id=hashlib.sha256(f"{identity}:{generation}:{ordinal}".encode()).hexdigest(),
                    document_id=identity,
                    ordinal=ordinal,
                    text=text,
                    title=document.title,
                    author=document.author,
                    section=unit.section,
                    page=unit.page,
                    line_start=bisect.bisect_left(line_breaks, start) + 1
                    if unit.page is None
                    else None,
                    line_end=bisect.bisect_left(line_breaks, end - 1) + 1
                    if unit.page is None
                    else None,
                    char_start=start,
                    char_end=end,
                    tokens=_tokens(text),
                    tags=tags,
                    concepts=_concepts(text),
                )
            )
    if not chunks:
        raise DocumentError("Source contains no indexable passages.")
    return document, chunks


class Ingestor:
    def __init__(self, settings: Settings, store: VectorStore, provider: ProviderClient) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider

    async def ingest(
        self,
        path: Path,
        title: str | None = None,
        author: str | None = None,
        rights: str = "user-authorized personal processing",
    ) -> IngestResult:
        document, chunks = await asyncio.to_thread(
            _prepare, path, self.settings, title, author, rights
        )
        existing = await self.store.get_document(document.id)
        if existing is not None and existing.generation == document.generation:
            return IngestResult(
                document_id=document.id,
                status="unchanged",
                chunk_count=len(chunks),
            )
        embeddings: list[list[float]] = []
        size = self.settings.embedding_batch_size
        for start in range(0, len(chunks), size):
            batch = chunks[start : start + size]
            vectors = await self.provider.embed([chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise ProviderError(
                    "Embedding response count does not match the requested passage count."
                )
            for vector in vectors:
                if (
                    len(vector) != self.settings.embedding_dimensions
                    or any(not math.isfinite(value) for value in vector)
                    or not 0 < math.hypot(*vector) < math.inf
                ):
                    raise ProviderError(
                        "Embedding response contains an invalid dimension, "
                        "nonfinite value, or zero vector."
                    )
            embeddings.extend(vectors)
        written = await self.store.upsert_document(document, chunks, embeddings)
        return IngestResult(
            document_id=document.id,
            status="indexed" if written else "unchanged",
            chunk_count=len(chunks),
        )
