# Mimir-RAG

A personal-library RAG engine for **Claude Code and Codex**. Ingest authorized PDF, UTF-8 TXT or Markdown; ask questions with hybrid search, checked evidence, precise citations and source-backed concept mapping.

The library lives in SQLite on your machine. OpenAI embeddings and OpenAI/Anthropic synthesis transmit text to cloud APIs. Claude Code and Codex subscriptions do not supply these API credentials automatically. The default runtime uses `text-embedding-3-small` with 1,536 dimensions and `gpt-4.1-mini`; both synthesis and verification model selection are configurable.

## Install the shared runtime

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required for the commands below:

```bash
uv tool install git+https://github.com/Seungwoo321/mimir-rag.git
mimir-rag doctor
```

Set `OPENAI_API_KEY` in the invoking process through your shell or OS secret store. Never commit it. Configuration comes from environment variables; `.env.example` lists nonsecret examples and is not automatically loaded. For Anthropic synthesis set `ANTHROPIC_API_KEY` and `MIMIR_SYNTHESIS_PROVIDER=anthropic`; omit `MIMIR_SYNTHESIS_MODEL` to select the Anthropic default, or choose a compatible structured-output model explicitly.

## Claude Code

Inside Claude Code:

```text
/plugin marketplace add Seungwoo321/mimir-rag
/plugin install mimir-rag@mimir-rag-marketplace
/mimir-rag:ingest /path/to/authorized-book.pdf
/mimir-rag:ask What does the source say about feedback loops?
```

For local development use `claude --plugin-dir /path/to/mimir-rag`. The plugin contains namespaced commands and the shared skill, with no always-running server or lifecycle hooks.

## Codex

Install the plugin through the repository marketplace:

```bash
codex plugin marketplace add Seungwoo321/mimir-rag
codex plugin add mimir-rag@mimir-rag-marketplace
```

Alternatively clone the repository and install only the shared skill:

```bash
git clone https://github.com/Seungwoo321/mimir-rag.git
cd mimir-rag
python3 scripts/install_skill.py
```

Restart Codex, then invoke `$mimir-rag` with an ingestion request or a question. The skill calls the installed shared CLI. Use `--target` to select a project skill directory or an older installation's skill path. The installer preserves existing differing content. Plugin installation does not silently install Python dependencies; install the runtime first.

## CLI

```bash
mimir-rag ingest ./book.pdf --title "Systems Handbook" --author "Example Author"
mimir-rag ask "How does the handbook describe feedback?" --json
mimir-rag list
mimir-rag graph
mimir-rag delete document-id
```

The Python CLI also accepts literal `/ingest` and `/ask` command aliases. `ask` returns exit 0 for an accepted answer, 3 for an explicit abstention and 2 for an actionable error. Citations use stored titles/sections and physical PDF pages or decoded-text line spans. Source links address the original local file; the database retains evidence even if that file subsequently moves or changes, and a link alone does not prove the current file matches the indexed generation.

`MIMIR_DB_PATH` or the global `--db` flag selects a library. The default is `$XDG_DATA_HOME/mimir-rag/library.sqlite3`, falling back to `~/.local/share/mimir-rag/library.sqlite3`. The immediate data directory must be private. Model/dimension changes require a new library and re-ingestion. Exact dense search has O(N × dimensions) cost and a default 50,000-chunk capacity; this is a bounded local library, not an approximate distributed index.

## Four-phase blueprint

| Phase | Complete artifact |
| --- | --- |
| 1. Architecture, topology, metadata, copyright boundaries | [Architecture](docs/architecture.md), [database schema](docs/schema.md) |
| 2. Complete asynchronous Python core | [config.py](src/mimir_rag/config.py), [ingestor.py](src/mimir_rag/ingestor.py), [vector_store.py](src/mimir_rag/vector_store.py), [generator.py](src/mimir_rag/generator.py), [main.py](src/mimir_rag/main.py) |
| 3. Exact synthesis and self-reflection prompts | [Prompts](docs/prompts.md), runtime constants in [generator.py](src/mimir_rag/generator.py) |
| 4. Failure mitigations and reproducible verification | [Verification playbook](docs/verification.md), [tests](tests) |

The generator accepts only structured claims with known evidence IDs and exact source excerpts, then requires a separate grounding-review call before rendering citations. This improves grounding but cannot guarantee semantic infallibility. Concept mentions and co-occurrence are distinguished from verified descriptive relationships. Output is abstractive and copying is bounded; these controls do not certify copyright permission or fair use. Only ingest sources you are authorized to process.

Image-only PDFs require external OCR. PDF extraction cannot reliably infer printed page numbers, missing authors or semantic reading order. The tokenizer vocabulary is bundled with a verified checksum, so parsing and offline tests do not download assets on first use. The local database is owner-protected plaintext, and deleting searchable rows is not forensic erasure.

## Develop and verify

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run python -m build
uv run python scripts/validate_package.py
claude plugin validate . --strict
```

All automated tests use synthetic sources and mocked provider transports. See the [playbook](docs/verification.md) for intentional live checks and recovery. Local deterministic checks are authoritative for this package; it has no redundant GitHub Actions workflow. License: [MIT](LICENSE). Bundled tokenizer attribution: [third-party notices](THIRD_PARTY_NOTICES.md).
