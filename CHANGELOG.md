# Changelog

## 0.1.1 — 2026-10-01

- Install the PDF font decoder required for embedded CFF Type1 character maps.
- Bind document generations to extracted text and citation coordinates so decoder changes refresh unchanged source files.
- Publish matching Python, Claude Code and Codex plugin versions.

Re-ingest existing documents to apply the extraction fingerprint. Previously indexed PDF text may require refreshed decoding; re-ingestion preserves each canonical source identity and replaces its passages atomically.
