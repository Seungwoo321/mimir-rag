# Grounded answer prompts and schemas

Runtime constants in [generator.py](../src/mimir_rag/generator.py) are the authoritative exact prompts: `INTERNAL_SYSTEM_PROMPT` for synthesis and `SELF_REFLECTION_PROMPT` for independent grounding review. Print them without invoking a model:

```sh
python -c 'from mimir_rag.generator import INTERNAL_SYSTEM_PROMPT; print(INTERNAL_SYSTEM_PROMPT, end="")'
python -c 'from mimir_rag.generator import SELF_REFLECTION_PROMPT; print(SELF_REFLECTION_PROMPT, end="")'
```

`AnswerProposal` contains `answerable`, atomic `claims`, and optional `connections`. Each item has a unique ID, plain text, and evidence entries with a retrieved `source_id` and exact contiguous `quote`. Connections have two source-anchored labels and relation `co_occurs` or `described_connection`. Both labels are anchored to the same cited passage; described relationships require explicit quotation support. Item limits apply to claims and connections.

`GroundingReview` contains `approved`, `claim_verdicts`, and `connection_verdicts`. Each verdict has only an item `id` and Boolean `supported`. Every item requires exactly one positive verdict. Unknown, duplicate, missing, negative or malformed verdicts cause abstention. Schemas forbid extra fields and runtime validation rejects coercion.

```sh
python -c 'import json; from mimir_rag.generator import AnswerProposal, GroundingReview; print(json.dumps({"synthesis": AnswerProposal.model_json_schema(), "verification": GroundingReview.model_json_schema()}, indent=2))'
```

## Agent workflow

Default `ask --json` uses two isolated tool-less Claude Code calls for synthesis and review. Its installed authenticated CLI supplies generation without an API key.

Explicit agent-mode `ask --json` and `retrieve --json` return a private retrieval session, evidence hash, prompts, schemas, and bounded payload. The host produces a proposal using that payload. `review-context --session ID --proposal FILE --evidence-hash HASH` validates the proposal and supplies the exact independent review envelope plus its canonical proposal hash. A separate reviewer returns an envelope with `evidence_hash`, `proposal_hash`, and `review`. `verify --session ID --proposal FILE --review FILE --evidence-hash HASH --json` checks both bindings, reloads current source chunks, validates quotation/citation coverage, checks verdicts, and renders accepted Markdown or abstention.

Reviewer independence is a host workflow obligation. Hashes bind data; they do not authenticate the reviewer or prove semantic entailment. The plugin does not invoke OpenAI models. Codex invocation still uses Claude Code for generation. Explicit agent mode requires non-OpenAI synthesis and review contexts. Optional explicitly selected Anthropic synthesis uses separate production and grounding-review calls.

The application renders citations before each factual claim and concept connection using stored title, section, physical page or decoded line span, source offsets, and generation-specific chunk ID. Canonical local `file:` URIs and escaped metadata prevent model-authored destinations and active Markdown injection.

The context budget covers serialized question, metadata/text, prompts, schemas, and review proposal. Oversized passages are skipped rather than truncated into uncertain provenance. Weak evidence, invalid output, stale session, failed review or exhausted budget produces abstention. Source token-window checks bound aggregate copying, including Korean and Japanese fragments. They reduce reproduction but establish no fair-use threshold or publication right; see [privacy boundaries](architecture.md#privacy-copyright-and-recovery).
