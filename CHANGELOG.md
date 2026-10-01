# Changelog

## 0.2.0 — 2026-10-02

- Replace cloud embeddings with pinned local multilingual E5 inference and complete bounded-window aggregation.
- Default to isolated authenticated Claude Code synthesis and grounding review with deterministic citation verification.
- Remove OpenAI inference and API-key requirements; retain explicit non-OpenAI agent workflow and Anthropic API synthesis.
- Bind embedding fingerprints to the complete local inference contract and reject incompatible libraries.

## 0.1.2 — 2026-10-02

- Support explicitly configured libraries up to 150,000 chunks while retaining the 50,000-chunk default.
- Preserve bounded dense-search batches and transactional capacity enforcement for larger complete-book collections.
- Verify retrieval beyond 50,000 chunks and atomic rejection when configured capacity is exceeded.

## 0.1.1 — 2026-10-01

- Install the PDF font decoder required for embedded CFF Type1 character maps.
- Bind document generations to extracted text and citation coordinates so decoder changes refresh unchanged source files.
- Publish matching Python, Claude Code and Codex plugin versions.

Re-ingest existing documents to apply the extraction fingerprint. Previously indexed PDF text may require refreshed decoding; re-ingestion preserves each canonical source identity and replaces its passages atomically.
