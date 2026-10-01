# Local library schema

SQLite is the authority for document metadata, source passages, embeddings, lexical indexes and concept mentions. Schema version `1` is stored in both `library_meta.schema_version` and `PRAGMA user_version`. The application validates existing databases read-only before changing permissions or enabling WAL. Unknown versions, incomplete metadata and corruption cause a preservation-first error; initialization never migrates or rebuilds an unknown library.

| Relation | Identity | Payload and constraints |
| --- | --- | --- |
| `library_meta` | `key` | Schema version, embedding fingerprint, dimensions and `float32-le-l2-v1` encoding |
| `documents` | Stable `id` | Source URI, content hash, generation, title, nullable author, rights, metadata origin, ingestion signature and update timestamp |
| `chunks` | Generation-specific `id` | Document FK, unique document/ordinal pair, text, title, author, section, physical page or decoded-text line span, exact character span, token count, JSON semantic tags and concepts, normalized embedding BLOB |
| `chunks_fts` | Chunk `rowid` | External-content FTS5 index over text, title, section and normalized concept labels |
| `concept_mentions` | Chunk/normalized-label pair | Source-backed concept occurrence with cascading chunk FK |

Chunk insert, update and delete triggers maintain FTS5 within the same transaction. Cascading document deletion removes chunks and concept mentions. Embedding BLOBs use normalized little-endian float32 values; the byte count is exactly four times the configured dimensions. Nonfinite, zero-norm, incorrectly dimensioned or corrupted normalized vectors are rejected.

Document replacement starts with `BEGIN IMMEDIATE` after every embedding has been prepared and validated. It checks schema, fingerprint and global capacity, removes the prior generation, inserts the complete replacement and commits once. An identical document generation is a no-op. Any insertion failure rolls back document metadata, chunks, FTS changes and concept mentions together. Cancellation observed before commit rolls back; a completed commit is the operation's linearization point.

Writers serialize through SQLite's bounded busy timeout. Search opens one read transaction covering both dense and lexical retrieval. Dense ranking scans bounded NumPy batches with exact cosine similarity. BM25 sorts in ascending order. Query terms are Unicode word tokens quoted as FTS literals. Reciprocal rank fusion sums `1 / (rrf_k + rank)` for each candidate's available rankings; chunk ID breaks ties. Ranking scores are ordering values, not calibrated probabilities.

Concept graphs contain concept and source-chunk nodes, `mentions` edges, and explicitly marked `co_occurrence` edges with their supporting chunk IDs. Co-occurrence does not assert causation, entailment or semantic agreement.

New database directories are created with owner-only permissions and the database with mode `600` on POSIX. Existing shared directories are rejected instead of changing arbitrary ancestor permissions. Symlink traversal and multiply linked database files are rejected. A complete temporary database is published without overwriting a concurrent initializer's file. The database contains plaintext source material; document deletion removes searchable records but does not guarantee forensic erasure from WAL pages or backups.

The public asynchronous API is `VectorStore(Settings)`, `initialize`, `get_document`, `has_chunks`, `document_chunk_count`, `upsert_document`, `search`, `list_documents`, `delete_document` and `concept_graph`. SQLite and NumPy work runs outside the event loop. `has_chunks` checks availability without loading source contents. Inspection and deletion do not make provider calls. The embedding fingerprint and document identity contracts are defined in [architecture.md](architecture.md).
