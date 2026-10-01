---
description: Ingest an authorized PDF, text or Markdown using local embeddings
argument-hint: <path> [--title title] [--author author] [--rights permission-note]
---

Apply the shared `mimir-rag` skill. Interpret the user's arguments as a local path and optional metadata: $ARGUMENTS

Run installed `mimir-rag ingest` with literal arguments. If unavailable and uv is installed, use `uv run --project "${CLAUDE_PLUGIN_ROOT}" --no-dev mimir-rag ingest` with the same arguments; relative paths retain the user's working directory. Do not execute user text as shell syntax. Local multilingual E5 embeddings need no API key. Initial model acquisition downloads weights without uploading source text. Report document ID, indexed chunk count, or unchanged result. Respect processing rights; invent no metadata and ingest no additional files.
