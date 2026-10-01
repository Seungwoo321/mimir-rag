---
description: Ingest an authorized PDF, text or Markdown source into your personal library
argument-hint: <path> [--title title] [--author author] [--rights permission-note]
---

Apply the shared `mimir-rag` skill. Interpret the user's arguments as a local source path and optional metadata: $ARGUMENTS

Run the installed `mimir-rag ingest` command with literal arguments. If the CLI is unavailable and uv is installed, run `uv run --project "${CLAUDE_PLUGIN_ROOT}" --no-dev mimir-rag ingest` with the same literal arguments. The project flag preserves relative source paths in the user's working directory. Preserve argument quoting; do not execute user text as shell syntax. Report the document ID, indexed chunk count, or unchanged result. Cloud embeddings send parsed source text to OpenAI; respect the user's processing authorization. Do not invent missing metadata or ingest additional files.
