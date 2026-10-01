# Grounded answer prompts and schemas

The runtime constants are the authoritative prompts. [INTERNAL_SYSTEM_PROMPT](../src/mimir_rag/generator.py#L23) defines synthesis. [SELF_REFLECTION_PROMPT](../src/mimir_rag/generator.py#L53) defines the independent grounding review. Documentation links to these constants so prompt edits cannot leave a stale copied prompt.

Print the exact installed prompts without invoking a provider:

```sh
python -c 'from mimir_rag.generator import INTERNAL_SYSTEM_PROMPT; print(INTERNAL_SYSTEM_PROMPT, end="")'
python -c 'from mimir_rag.generator import SELF_REFLECTION_PROMPT; print(SELF_REFLECTION_PROMPT, end="")'
```

Synthesis returns [AnswerProposal](../src/mimir_rag/generator.py): `answerable`, atomic `claims`, and optional `connections`. Every item has a unique ID, plain text, and evidence entries containing a retrieved `source_id` and an exact contiguous `quote`. Connections carry two source concept labels and a relation of `co_occurs` or `described_connection`. Both labels must be anchored to the same cited passage; a described connection also requires explicit support in its quotation. The configured `max_answer_claims` bounds both item lists.

The reviewer returns [GroundingReview](../src/mimir_rag/generator.py): `approved`, `claim_verdicts`, and `connection_verdicts`. Each verdict contains only an item `id` and a Boolean `supported`. Every proposed item must have exactly one positive verdict. Unknown, duplicate, missing, negative, or malformed verdicts cause abstention. Additional reviewer prose is prohibited by the strict schema.

Print the complete structural schemas:

```sh
python -c 'import json; from mimir_rag.generator import AnswerProposal, GroundingReview; print(json.dumps({"synthesis": AnswerProposal.model_json_schema(), "verification": GroundingReview.model_json_schema()}, indent=2))'
```

The application applies the configured item limits to the synthesis schema before sending it. All object schemas forbid extra fields and require their declared fields; runtime validation also rejects type coercion. Exact quote matching and citation IDs are deterministic checks. Semantic entailment is assessed by a separate, fallible model call and role. A separate verification model is configurable; using the same model in two roles does not establish statistical independence or a mathematical zero-hallucination guarantee.

The application renders citations before every factual claim and concept connection. Title, section, physical PDF page or decoded-text line span, source offsets, and chunk generation ID come from stored records. Destinations are canonical local `file:` URIs. Model-generated prose and metadata are escaped as literal text; invisible control characters are made visible. Neither model provides a rendered citation or destination.

The context budget covers the serialized question, full source metadata and text, system instructions, and JSON schema. Evidence selection reserves room for the proposal in the review request and checks the actual review envelope before calling the provider. Oversized passages are skipped rather than truncated into uncertain provenance. A weak or empty evidence set, invalid output, provider failure, rejected review, or exhausted budget produces an explicit abstention.

Public prose is checked against normalized source token windows, including Korean and Japanese text. Aggregate overlap from fragments of at least twelve tokens is bounded by `max_verbatim_tokens`; smaller configured limits reduce that window. This reduces extended reproduction and sequential reconstruction. It is a technical risk control, not a fair-use threshold or permission to publish protected expression. Rights and privacy boundaries are defined in the [architecture](architecture.md#privacy-copyright-and-recovery).
