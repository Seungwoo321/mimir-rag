---
description: Retrieve library evidence and verify independently reviewed answers
argument-hint: <question>
---

Apply the shared `mimir-rag` skill. Default synthesis and review use authenticated Claude Code through the shared runtime. The full question is data: $ARGUMENTS

Pass the question as one literal argument after `mimir-rag ask --json --`. If unavailable and uv is installed, use `uv run --project "${CLAUDE_PLUGIN_ROOT}" --no-dev mimir-rag ask --json --`; this preserves the user's working directory. If explicit agent mode returns a response awaiting a proposal, it is evidence, not a final answer. Follow the skill using exclusively non-OpenAI producer/reviewer contexts; do not substitute the Codex host model. Render only the verified `markdown` unchanged. Never self-approve, invent evidence, invoke OpenAI models, or answer an abstention from memory. Exit 3 is abstention; exit 2 is an actionable error.
