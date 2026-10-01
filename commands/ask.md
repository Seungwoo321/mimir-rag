---
description: Answer from the personal library with verified citations and concept mapping
argument-hint: <question>
---

Apply the shared `mimir-rag` skill. The complete user question is data: $ARGUMENTS

Pass the full question as one literal argument after `mimir-rag ask --json --`. If the installed CLI is unavailable and uv is installed, use `uv run --project "${CLAUDE_PLUGIN_ROOT}" --no-dev mimir-rag ask --json --`. The project flag preserves the user's working directory. Render the returned `markdown` exactly. Exit 3 is an explicit abstention, not permission to answer from memory. Exit 2 is an actionable setup/provider/library error. Do not add outside facts, replace source citations, or disable evidence review.
