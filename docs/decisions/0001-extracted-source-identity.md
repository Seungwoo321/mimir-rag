# Extracted content belongs to document generation

Status: Accepted

Date: 2026-10-01

## Context

A PDF decoder can extract different Unicode text from unchanged source bytes. A raw-file hash therefore cannot determine whether indexed passages and citation coordinates remain identical. Embedded CFF Type1 fonts also require the declared PDF fonts dependency.

## Decision

Keep the canonical source URI as the stable document identity and retain the raw-file SHA256 as source provenance. Include a separate SHA256 of the indexed extracted units in the generation payload. Each canonically framed unit contains its text slice, character bounds, line bounds, section and physical page. Derive line bounds from the same newline offsets used by citations, including excluded prefixes. Hash only its indexed slice to avoid repeatedly hashing a complete document for each section.

Declare `pypdf[fonts]` so a fresh installation includes the CFF decoder. Generation equality depends on actual parsed output and configuration. It does not depend on a decoder version label.

## Alternatives

| Choice | Effect |
| --- | --- |
| Raw source hash alone | Misses changed glyph decoding or citation coordinates when the file is unchanged. |
| Dependency version in generation | Refreshes unchanged output after a version change and does not directly identify the indexed content. |
| Extracted-unit fingerprint | Detects changed text or citation coordinates while preserving a no-op for identical parsed output. |

The extracted-unit fingerprint applies content-addressability to the passages that citations actually identify. It preserves the separation between source identity, raw provenance and searchable generation.

## Consequences

Re-ingestion replaces changed passages atomically under the existing document identity. Identical parsed output avoids another embedding call. The database schema and embedding-space fingerprint remain unchanged. Existing documents acquire the extracted-content fingerprint when re-ingested; upgrading does not send stored documents to a provider automatically.

Synthetic CFF and same-byte extraction regressions exercise this contract against the real local store in [test_extraction_identity.py](../../tests/test_extraction_identity.py).
