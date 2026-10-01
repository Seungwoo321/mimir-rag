# Verification and edge-case playbook

## Local verification

Run from the repository root:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run python -m build
uv run python scripts/validate_package.py
claude plugin validate .claude-plugin/marketplace.json --strict
claude plugin validate .claude-plugin/plugin.json --strict
claude plugin validate commands --strict
claude plugin validate skills --strict
```

Tests use synthetic documents and deterministic embedding/synthesis doubles. Provider HTTP tests use mock transports and validate actual request payloads, timeout retries, refusal handling, and output decoding. They never read a user's books, discover secrets, or call a paid endpoint. A passing offline suite establishes application contracts, not current provider availability or model grounding accuracy on arbitrary books.

For intentional live verification, install the CLI, set `OPENAI_API_KEY` in the process environment, ingest a synthetic licensed text, and ask a question whose answer can be manually checked. Anthropic synthesis additionally needs `ANTHROPIC_API_KEY` and `MIMIR_SYNTHESIS_PROVIDER=anthropic`. Never save keys in a transcript. Evaluate supported and unsupported questions, adversarial document instructions, and citation accuracy independently before relying on generated answers for consequential decisions.

## Three principal failure modes

| Failure | Observable behavior | Code-level mitigation | Verification |
| --- | --- | --- | --- |
| Out-of-context question or unsupported claim | Search finds weak evidence, synthesis declares unanswerable, or reviewer cannot prove every claim | `generator.py` applies a cosine floor and context budget, validates IDs and exact excerpts, requires complete positive reviewer coverage, and emits an explicit abstention | Unrelated question, empty corpus, invented source ID, unsupported relation, missing review decision, prompt injection, and provider refusal |
| Embedding API timeout or rate limit | Provider raises a bounded failure while collecting a new document generation | `providers.py` retries only transient errors with bounded exponential backoff and bounded `Retry-After`; `ingestor.py` completes all embeddings before `vector_store.py` atomically replaces rows | Timeout/429/503 mocked responses, exhausted retries, mid-batch failure, and cancellation all preserve the prior searchable generation |
| Semantic drift from a model or dimension change | A library is queried or written using incompatible vectors | `config.py` defines the embedding-space fingerprint; `vector_store.py` checks it before writes/searches and rejects mismatches and invalid vectors | Same dimensions with different model, different dimensions, NaN/Infinity, zero vector, and corrupted stored vector |

## Additional boundaries

- Image-only, encrypted, malformed, oversized, and excessive-page PDFs fail with actionable errors; OCR is an explicit preprocessing operation outside the parser.
- UTF-8 text offsets and physical PDF page offsets are checked against the immutable parsed input. Chunk overlap cannot cross a section or page boundary.
- Identical ingestion is a no-op. Changed content and metadata replace one stable document identity. Concurrent writers cannot expose partial versions.
- FTS insert/update/delete behavior is checked after replacement and deletion. SQL-like punctuation in a question remains data.
- Future or incomplete schemas fail without overwrite. Database operations report lock/corruption errors without printing source contents.
- Titles, author fields, section labels, claims, and concept labels cannot inject active Markdown/HTML or arbitrary links into rendered answers.
- Concept co-occurrence is labeled as co-occurrence; causal/supportive prose requires checked evidence and a positive grounding review.
- Owner-only persistence and Git ignore rules protect local artifacts. Deletion removes searchable records but does not claim secure erasure from backups or old WAL pages.
- Verbatim-copy checks and concise abstractive output reduce reproduction; they do not determine whether a particular use is legally permitted.

## Operational recovery

Keep a filesystem or SQLite online-backup snapshot before deliberate library maintenance. Restore only by a user-authorized operation; automatic ingestion failure uses transaction rollback rather than replacing the database. To change embedding models, select a new `MIMIR_DB_PATH` and re-ingest the authorized corpus. Preserve the old library until the new one has passed independent retrieval checks.

The library is an owner-protected plaintext database. Full-disk encryption and secure backups are operating-system responsibilities. Do not publish source documents, vector databases, private paths, environment files, or live API transcripts with the plugin source.
