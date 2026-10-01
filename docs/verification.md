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

For genuine local-weight verification, ingest a synthetic licensed text with no API credentials, inspect retrieved passages, and use authenticated Claude Code for independently reviewed answers. For explicit agent mode, complete proposal/review/verify using non-OpenAI contexts. Check multilingual retrieval and long inputs beyond one encoder window. Optional Anthropic synthesis needs `ANTHROPIC_API_KEY` and `MIMIR_SYNTHESIS_PROVIDER=anthropic`. Never save keys in a transcript. Evaluate supported and unsupported questions, adversarial document instructions, and citation accuracy independently before relying on generated answers for consequential decisions.

## Three principal failure modes

| Failure | Observable behavior | Code-level mitigation | Verification |
| --- | --- | --- | --- |
| Out-of-context question or unsupported claim | Search finds weak evidence, synthesis declares unanswerable, or reviewer cannot prove every claim | `generator.py` applies a cosine floor and context budget, validates IDs and exact excerpts, requires complete positive reviewer coverage, and emits an explicit abstention | Unrelated question, empty corpus, invented source ID, unsupported relation, missing review decision, prompt injection, and provider refusal |
| Local model/cache or inference failure | Weights are unavailable, device setup fails, or a new generation cannot be fully embedded | `providers.py` uses pinned safe weights and bounded complete windows; `ingestor.py` finishes valid embeddings before `vector_store.py` atomically replaces rows | Missing offline cache, incompatible output, long-tail coverage, mid-batch failure and cancellation preserve the prior searchable generation |
| Semantic drift from a model or dimension change | A library is queried or written using incompatible vectors | `config.py` defines the embedding-space fingerprint; `vector_store.py` checks it before writes/searches and rejects mismatches and invalid vectors | Changed revision/window/pooling contract, incompatible dimensions, NaN/Infinity, zero vector, and corrupted stored vector |

## Agent session verification

Check wrong evidence hashes, changed proposal hashes, forged source IDs, invalid exact quotations, rejected/missing/duplicate review verdicts, malformed envelopes, stale sessions after replacement/deletion, and private session paths. A separate reviewer inspects source entailment; deterministic hashes do not prove semantic truth or reviewer independence. A valid proposal never becomes an answer before `verify` succeeds. No test or runtime path invokes an OpenAI model.

## Additional boundaries

- Image-only, encrypted, malformed, oversized, and excessive-page PDFs fail with actionable errors; OCR is an explicit preprocessing operation outside the parser.
- UTF-8 text offsets and physical PDF page offsets are checked against the immutable parsed input. Chunk overlap cannot cross a section or page boundary.
- Identical ingestion is a no-op. Changed content, extracted text or citation coordinates, and metadata replace one stable document identity. Synthetic embedded CFF fonts verify Unicode decoding and same-byte decoder upgrades with a real SQLite store. Concurrent writers cannot expose partial versions.
- FTS insert/update/delete behavior is checked after replacement and deletion. SQL-like punctuation in a question remains data.
- Future or incomplete schemas fail without overwrite. Database operations report lock/corruption errors without printing source contents.
- Titles, author fields, section labels, claims, and concept labels cannot inject active Markdown/HTML or arbitrary links into rendered answers.
- Concept co-occurrence is labeled as co-occurrence; causal/supportive prose requires checked evidence and a positive grounding review.
- Owner-only persistence and Git ignore rules protect local artifacts. Deletion removes searchable records but does not claim secure erasure from backups or old WAL pages.
- Verbatim-copy checks and concise abstractive output reduce reproduction; they do not determine whether a particular use is legally permitted.

## Operational recovery

Keep a filesystem or SQLite online-backup snapshot before deliberate library maintenance. Restore only by a user-authorized operation; automatic ingestion failure uses transaction rollback rather than replacing the database. To change embedding models, select a new `MIMIR_DB_PATH` and re-ingest the authorized corpus. Preserve the old library until the new one has passed independent retrieval checks.

The library is an owner-protected plaintext database. Full-disk encryption and secure backups are operating-system responsibilities. Do not publish source documents, vector databases, private paths, environment files, or live API transcripts with the plugin source.
