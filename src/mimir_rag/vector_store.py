from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import stat
import tempfile
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import closing
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from .config import Settings
from .errors import EmbeddingSpaceError, StoreError
from .models import Chunk, Document, SearchHit

SCHEMA_VERSION = 1
_T = TypeVar("_T")
_INITIALIZATION_LOCKS: dict[Path, threading.Lock] = {}
_LOCK_REGISTRY = threading.Lock()
_DOCUMENT_FIELDS = tuple(Document.model_fields)
_CHUNK_FIELDS = tuple(Chunk.model_fields)
_SCHEMA = (
    "CREATE TABLE library_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE documents (
        id TEXT PRIMARY KEY, source_uri TEXT NOT NULL, content_hash TEXT NOT NULL,
        generation TEXT NOT NULL, title TEXT NOT NULL, author TEXT,
        rights TEXT NOT NULL, metadata_origin TEXT NOT NULL, ingest_signature TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )""",
    """CREATE TABLE chunks (
        id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0), text TEXT NOT NULL,
        title TEXT NOT NULL, author TEXT, section TEXT NOT NULL, page INTEGER,
        line_start INTEGER, line_end INTEGER, char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL, tokens INTEGER NOT NULL CHECK (tokens > 0),
        tags TEXT NOT NULL, concepts TEXT NOT NULL, concepts_text TEXT NOT NULL,
        embedding BLOB NOT NULL, UNIQUE(document_id, ordinal)
    )""",
    """CREATE TABLE concept_mentions (
        chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        label TEXT NOT NULL, PRIMARY KEY(chunk_id, label)
    )""",
    "CREATE INDEX chunks_document ON chunks(document_id)",
    "CREATE INDEX concept_label ON concept_mentions(label)",
    """CREATE VIRTUAL TABLE chunks_fts USING fts5(
        text, title, section, concepts_text, content='chunks', content_rowid='rowid'
    )""",
    """CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text, title, section, concepts_text)
        VALUES(new.rowid, new.text, new.title, new.section, new.concepts_text);
    END""",
    """CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, title, section, concepts_text)
        VALUES('delete', old.rowid, old.text, old.title, old.section, old.concepts_text);
    END""",
    """CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, title, section, concepts_text)
        VALUES('delete', old.rowid, old.text, old.title, old.section, old.concepts_text);
        INSERT INTO chunks_fts(rowid, text, title, section, concepts_text)
        VALUES(new.rowid, new.text, new.title, new.section, new.concepts_text);
    END""",
)


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.db_path
        self._initialized = False

    async def _run(self, operation: Callable[[], _T]) -> _T:
        def guarded() -> _T:
            try:
                return operation()
            except StoreError:
                raise
            except sqlite3.Error as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise StoreError(
                        "Library is busy; retry after the current writer finishes."
                    ) from None
                raise StoreError(
                    "Database operation failed; preserve the library and inspect its integrity."
                ) from None
            except (OSError, ValidationError, ValueError, TypeError, OverflowError):
                raise StoreError(
                    "Library data or filesystem access is invalid; preserve it for recovery."
                ) from None

        return await asyncio.to_thread(guarded)

    async def initialize(self) -> None:
        await self._run(self._initialize)

    def _check_path(self) -> None:
        for path in (self.path, *self.path.parents):
            if path.is_symlink():
                raise StoreError("Database paths must not traverse symbolic links.")
        parent = self.path.parent
        missing: list[Path] = []
        cursor = parent
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                if not directory.is_dir() or directory.is_symlink():
                    raise StoreError(
                        "Database parent path changed during initialization."
                    ) from None
        info = parent.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise StoreError("Choose a dedicated directory for the database.")
        if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise StoreError(
                "Database directory must be owned by you with mode 700; "
                "choose a private directory instead of a shared parent."
            )
        if self.path.exists():
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StoreError("Database must be a regular file without extra hard links.")
            if os.name == "posix" and info.st_uid != os.getuid():
                raise StoreError("Database must be owned by the current user.")

    def _connect(self, mode: str = "rw") -> sqlite3.Connection:
        self._check_path()
        conn = sqlite3.connect(
            self.path.as_uri() + f"?mode={mode}",
            uri=True,
            timeout=self.settings.sqlite_busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self.settings.sqlite_busy_timeout_ms}")
        return conn

    def _metadata(self) -> dict[str, str]:
        return {
            "schema_version": str(SCHEMA_VERSION),
            "embedding_fingerprint": self.settings.embedding_fingerprint,
            "embedding_dimensions": str(self.settings.embedding_dimensions),
            "vector_encoding": "float32-le-l2-v1",
        }

    def _validate(self, conn: sqlite3.Connection, integrity: bool = False) -> None:
        expected_columns = {
            "library_meta": {"key", "value"},
            "documents": set(_DOCUMENT_FIELDS) | {"updated_at"},
            "chunks": set(_CHUNK_FIELDS) | {"concepts_text", "embedding"},
            "concept_mentions": {"chunk_id", "label"},
        }
        for table, expected in expected_columns.items():
            actual = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not expected <= actual:
                raise StoreError("Unsupported or incomplete library schema; use a new database.")
        objects = dict(conn.execute("SELECT name, sql FROM sqlite_master"))
        for statement in _SCHEMA:
            match = re.match(r"CREATE(?: VIRTUAL)? (?:TABLE|INDEX|TRIGGER) (\w+)", statement)
            if match is None:
                raise StoreError("Application schema definition is invalid.")
            actual_sql = objects.get(match.group(1))
            expected_ddl = re.sub(r"\s+", " ", statement).strip().rstrip(";").casefold()
            actual_ddl = re.sub(r"\s+", " ", actual_sql or "").strip().rstrip(";").casefold()
            if actual_ddl != expected_ddl:
                raise StoreError(
                    "Library schema objects are unknown or incomplete; preserve it for recovery."
                )
        metadata = dict(conn.execute("SELECT key, value FROM library_meta"))
        if metadata.get("schema_version") != str(SCHEMA_VERSION):
            raise StoreError(
                "Unsupported library schema version; preserve it and use a new database."
            )
        if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise StoreError("Library schema version is inconsistent; preserve it for recovery.")
        if any(key not in metadata for key in self._metadata()):
            raise StoreError("Library embedding metadata is missing; use a new database.")
        if any(metadata[key] != value for key, value in self._metadata().items()):
            raise EmbeddingSpaceError(
                "Embedding provider/model/dimensions differ from this library; "
                "select a new MIMIR_DB_PATH and re-ingest the authorized sources."
            )
        if integrity and conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise StoreError("Library integrity check failed; preserve it for recovery.")

    def _create(self) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".mimir-initialize-", dir=self.path.parent)
        temporary = Path(name)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(temporary, isolation_level=None)) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                    conn.executemany(
                        "INSERT INTO library_meta(key, value) VALUES (?, ?)",
                        self._metadata().items(),
                    )
                    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            # Atomic publication preserves a concurrent initializer's complete database.
            try:
                os.link(temporary, self.path, follow_symlinks=False)
            except FileExistsError:
                if self.path.is_symlink():
                    raise StoreError("Database path changed during initialization.") from None
        finally:
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + "-journal").unlink(missing_ok=True)

    def _initialize(self) -> None:
        with _LOCK_REGISTRY:
            lock = _INITIALIZATION_LOCKS.setdefault(self.path, threading.Lock())
        with lock:
            if self._initialized:
                return
            self._check_path()
            if not self.path.exists():
                self._create()
            # Inspect an existing file read-only before permissions or persistent WAL configuration.
            with closing(self._connect("ro")) as conn:
                self._validate(conn, integrity=True)
            if os.name == "posix":
                self.path.chmod(0o600)
            with closing(self._connect()) as conn:
                mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode != "wal":
                    raise StoreError("Library requires SQLite WAL mode for concurrent access.")
            self._initialized = True

    def _normalize(self, vector: Sequence[float]) -> NDArray[np.float32]:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.settings.embedding_dimensions,) or not np.isfinite(values).all():
            raise EmbeddingSpaceError("Embedding has invalid dimensions or nonfinite values.")
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            raise EmbeddingSpaceError("Embedding must have a nonzero norm.")
        values = values / scale
        return np.asarray(values / np.linalg.norm(values), dtype="<f4")

    def _decode(self, blob: bytes) -> NDArray[np.float32]:
        if len(blob) != 4 * self.settings.embedding_dimensions:
            raise EmbeddingSpaceError("Stored vector dimensions are corrupt; preserve the library.")
        vector = np.frombuffer(blob, dtype="<f4")
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise EmbeddingSpaceError(
                "Stored vector normalization is corrupt; preserve the library."
            )
        return vector

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        return Document.model_validate({field: row[field] for field in _DOCUMENT_FIELDS})

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        values = {field: row[field] for field in _CHUNK_FIELDS}
        values["tags"] = json.loads(values["tags"])
        values["concepts"] = json.loads(values["concepts"])
        return Chunk.model_validate(values)

    async def get_document(self, document_id: str) -> Document | None:
        await self.initialize()

        def read() -> Document | None:
            with closing(self._connect("ro")) as conn:
                row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
                return None if row is None else self._document(row)

        return await self._run(read)

    async def has_chunks(self) -> bool:
        await self.initialize()

        def read() -> bool:
            with closing(self._connect("ro")) as conn:
                self._validate(conn)
                return bool(conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()[0])

        return await self._run(read)

    async def document_chunk_count(self, document_id: str) -> int:
        await self.initialize()

        def read() -> int:
            with closing(self._connect("ro")) as conn:
                return int(
                    conn.execute(
                        "SELECT count(*) FROM chunks WHERE document_id=?", (document_id,)
                    ).fetchone()[0]
                )

        return await self._run(read)

    async def upsert_document(
        self, document: Document, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> bool:
        await self.initialize()
        cancelled = threading.Event()
        operation = asyncio.create_task(
            self._run(lambda: self._upsert(document, chunks, embeddings, cancelled))
        )
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await asyncio.shield(operation)
            except StoreError:
                pass
            raise

    def _upsert(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        cancelled: threading.Event,
    ) -> bool:
        if not chunks or len(chunks) != len(embeddings):
            raise StoreError("A document requires one valid embedding for each nonempty chunk.")
        if {chunk.ordinal for chunk in chunks} != set(range(len(chunks))):
            raise StoreError("Document chunk ordinals must be unique and contiguous.")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise StoreError("Document chunk identifiers must be unique.")
        if any(chunk.document_id != document.id for chunk in chunks):
            raise StoreError("A document transaction cannot contain another document's chunks.")
        vectors = [self._normalize(vector).tobytes() for vector in embeddings]
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._validate(conn)
                old = conn.execute("SELECT * FROM documents WHERE id=?", (document.id,)).fetchone()
                if old is not None and old["generation"] == document.generation:
                    if self._document(old) != document:
                        raise StoreError(
                            "A generation identifier cannot describe different metadata."
                        )
                    conn.execute("ROLLBACK")
                    return False
                total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                previous = conn.execute(
                    "SELECT count(*) FROM chunks WHERE document_id=?", (document.id,)
                ).fetchone()[0]
                if total - previous + len(chunks) > self.settings.max_chunks:
                    raise StoreError(
                        "Library chunk capacity reached; remove sources or use a new library."
                    )
                conn.execute("DELETE FROM documents WHERE id=?", (document.id,))
                fields = ", ".join(_DOCUMENT_FIELDS)
                placeholders = ", ".join("?" for _ in _DOCUMENT_FIELDS)
                conn.execute(
                    f"INSERT INTO documents({fields}) VALUES ({placeholders})",
                    tuple(getattr(document, field) for field in _DOCUMENT_FIELDS),
                )
                fields = ", ".join((*_CHUNK_FIELDS, "concepts_text", "embedding"))
                placeholders = ", ".join(
                    "?" for _ in (*_CHUNK_FIELDS, "concepts_text", "embedding")
                )
                for chunk, vector in zip(chunks, vectors, strict=True):
                    if cancelled.is_set():
                        raise StoreError("Document replacement was cancelled before commit.")
                    concepts = sorted(
                        {
                            " ".join(label.casefold().split())
                            for label in chunk.concepts
                            if label.strip()
                        }
                    )
                    values = chunk.model_dump()
                    values["tags"] = json.dumps(chunk.tags, ensure_ascii=False)
                    values["concepts"] = json.dumps(chunk.concepts, ensure_ascii=False)
                    conn.execute(
                        f"INSERT INTO chunks({fields}) VALUES ({placeholders})",
                        tuple(values[field] for field in _CHUNK_FIELDS)
                        + (" ".join(concepts), vector),
                    )
                    conn.executemany(
                        "INSERT INTO concept_mentions(chunk_id, label) VALUES (?, ?)",
                        [(chunk.id, label) for label in concepts],
                    )
                if cancelled.is_set():
                    raise StoreError("Document replacement was cancelled before commit.")
                conn.execute("COMMIT")
                return True
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _lexical_query(query: str) -> str:
        terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold(), flags=re.UNICODE)))[:32]
        return " OR ".join('"' + term + '"' for term in terms)

    async def search(
        self, query: str, embedding: Sequence[float], top_k: int | None = None
    ) -> list[SearchHit]:
        await self.initialize()
        limit = self.settings.top_k if top_k is None else top_k
        if not 1 <= limit <= 50:
            raise StoreError("Retrieval top_k must be between 1 and 50.")
        return await self._run(lambda: self._search(query, embedding, limit))

    def _search(self, query: str, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        vector = self._normalize(embedding)
        candidate_limit = limit * self.settings.candidate_multiplier
        with closing(self._connect("ro")) as conn:
            conn.execute("BEGIN")
            try:
                self._validate(conn)
                dense: list[tuple[float, sqlite3.Row]] = []
                cursor = conn.execute(
                    "SELECT c.*, d.source_uri FROM chunks c "
                    "JOIN documents d ON d.id=c.document_id ORDER BY c.id"
                )
                while rows := cursor.fetchmany(self.settings.dense_batch_size):
                    matrix = np.stack([self._decode(row["embedding"]) for row in rows])
                    scores = np.clip(matrix @ vector, -1.0, 1.0)
                    dense.extend(
                        (float(score), row) for score, row in zip(scores, rows, strict=True)
                    )
                    dense.sort(key=lambda item: (-item[0], item[1]["id"]))
                    del dense[candidate_limit:]
                lexical: list[sqlite3.Row] = []
                expression = self._lexical_query(query)
                if expression:
                    lexical = conn.execute(
                        "SELECT c.*, d.source_uri, bm25(chunks_fts) AS lexical_score "
                        "FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid "
                        "JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ? "
                        "ORDER BY lexical_score ASC, c.id ASC LIMIT ?",
                        (expression, candidate_limit),
                    ).fetchall()
                fused: dict[str, float] = defaultdict(float)
                records: dict[str, sqlite3.Row] = {}
                dense_scores: dict[str, float] = {}
                lexical_ranks: dict[str, int] = {}
                for rank, (score, row) in enumerate(dense, start=1):
                    key = row["id"]
                    records[key] = row
                    dense_scores[key] = score
                    fused[key] += 1 / (self.settings.rrf_k + rank)
                for rank, row in enumerate(lexical, start=1):
                    key = row["id"]
                    records[key] = row
                    dense_scores.setdefault(
                        key, float(np.clip(self._decode(row["embedding"]) @ vector, -1.0, 1.0))
                    )
                    lexical_ranks[key] = rank
                    fused[key] += 1 / (self.settings.rrf_k + rank)
                ranked = sorted(fused, key=lambda key: (-fused[key], key))[:limit]
                result = [
                    SearchHit(
                        chunk=self._chunk(records[key]),
                        score=fused[key],
                        dense_score=dense_scores[key],
                        lexical_rank=lexical_ranks.get(key),
                        source_uri=records[key]["source_uri"],
                    )
                    for key in ranked
                ]
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    async def list_documents(self) -> list[Document]:
        await self.initialize()

        def read() -> list[Document]:
            with closing(self._connect("ro")) as conn:
                return [
                    self._document(row)
                    for row in conn.execute(
                        "SELECT * FROM documents ORDER BY title COLLATE NOCASE, id"
                    )
                ]

        return await self._run(read)

    async def delete_document(self, document_id: str) -> bool:
        await self.initialize()

        def delete() -> bool:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    self._validate(conn)
                    removed = conn.execute(
                        "DELETE FROM documents WHERE id=?", (document_id,)
                    ).rowcount
                    conn.execute("COMMIT")
                    return bool(removed)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise

        return await self._run(delete)

    async def concept_graph(self, document_id: str | None = None) -> dict[str, object]:
        await self.initialize()

        def read() -> dict[str, object]:
            with closing(self._connect("ro")) as conn:
                statement = (
                    "SELECT m.label, c.id, c.document_id, d.source_uri FROM concept_mentions m "
                    "JOIN chunks c ON c.id=m.chunk_id JOIN documents d ON d.id=c.document_id"
                )
                parameters: tuple[str, ...] = ()
                if document_id is not None:
                    statement += " WHERE c.document_id=?"
                    parameters = (document_id,)
                rows = conn.execute(statement + " ORDER BY c.id, m.label", parameters).fetchall()
                nodes: dict[str, dict[str, str]] = {}
                edges: list[dict[str, Any]] = []
                mentions: dict[str, list[str]] = defaultdict(list)
                for row in rows:
                    concept, source = "concept:" + row["label"], "chunk:" + row["id"]
                    nodes[concept] = {"id": concept, "type": "concept", "label": row["label"]}
                    nodes[source] = {
                        "id": source,
                        "type": "chunk",
                        "document_id": row["document_id"],
                        "source_uri": row["source_uri"],
                    }
                    edges.append(
                        {
                            "source": concept,
                            "target": source,
                            "type": "mentions",
                            "chunk_id": row["id"],
                            "document_id": row["document_id"],
                        }
                    )
                    mentions[row["id"]].append(concept)
                pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
                for chunk_id, concepts in mentions.items():
                    for pair in combinations(concepts, 2):
                        pairs[pair].append(chunk_id)
                for (source, target), chunk_ids in sorted(pairs.items()):
                    edges.append(
                        {
                            "source": source,
                            "target": target,
                            "type": "co_occurrence",
                            "chunk_ids": chunk_ids,
                        }
                    )
                return {"nodes": [nodes[key] for key in sorted(nodes)], "edges": edges}

        return await self._run(read)
