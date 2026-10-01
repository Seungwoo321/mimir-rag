from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .errors import GroundingError
from .generator import (
    INTERNAL_SYSTEM_PROMPT,
    SELF_REFLECTION_PROMPT,
    AnswerProposal,
    Generator,
    GroundingReview,
    _json,
)
from .models import Answer, SearchHit

MAX_JSON_BYTES = 4_000_000
SESSION_TTL_SECONDS = 86_400


def _digest(value: object) -> str:
    try:
        return hashlib.sha256(_json(value).encode()).hexdigest()
    except (ValueError, TypeError) as exc:
        raise GroundingError("Evidence contains invalid JSON text.") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key.")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError("Nonfinite JSON number.")


def read_private_json(path: Path) -> Any:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise GroundingError("Evidence JSON must be a private regular file (mode 600).")
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise GroundingError("Evidence JSON exceeds its bounded file size.")
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 32:
                raise ValueError("JSON nesting exceeds its depth limit.")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        _json(value).encode()
        return value
    except (OSError, ValueError, RecursionError) as exc:
        raise GroundingError("Cannot read a bounded private evidence JSON file.") from exc


class EvidenceSessions:
    """Local integrity binding; a user-owned file is not authenticated agent identity."""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.directory = generator.settings.db_path.parent / "evidence-sessions"

    def _binding(self) -> str:
        return _digest(self.generator.settings.model_dump(mode="json"))

    def _write(self, session_id: str, record: dict[str, Any]) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
            raise GroundingError("Evidence session directory must be private (mode 700).")
        if len(list(self.directory.glob("*.json"))) >= 128:
            raise GroundingError("Evidence session capacity reached; remove completed sessions.")
        raw = _json(record).encode()
        if len(raw) > MAX_JSON_BYTES:
            raise GroundingError("Evidence session exceeds its bounded file size.")
        path = self.directory / f"{session_id}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)

    async def retrieve(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        question, hits, reason = await self.generator.retrieve(question, top_k)
        if reason:
            return {"status": "abstained", **self.generator._abstain(reason).model_dump()}
        session_id = uuid.uuid4().hex
        record = {
            "version": 1,
            "created_at": time.time(),
            "binding": self._binding(),
            "question": question,
            "hits": [hit.model_dump(mode="json") for hit in hits],
        }
        evidence_hash = _digest(record)
        await asyncio.to_thread(self._write, session_id, record)
        return {
            "status": "awaiting_proposal",
            "abstained": False,
            "session_id": session_id,
            "evidence_hash": evidence_hash,
            "synthesis_system_prompt": INTERNAL_SYSTEM_PROMPT,
            "verification_system_prompt": SELF_REFLECTION_PROMPT,
            "proposal_schema": self.generator._synthesis_schema(),
            "review_schema": GroundingReview.model_json_schema(),
            "payload": self.generator._payload(question, hits),
            "sources": [
                {
                    "source_id": hit.chunk.id,
                    "source_uri": hit.source_uri,
                    **hit.chunk.model_dump(exclude={"id", "text", "tokens", "tags", "concepts"}),
                }
                for hit in hits
            ],
            "verification": "Awaiting host proposal and independently produced semantic review.",
        }

    async def _load(self, session_id: str, evidence_hash: str) -> tuple[str, list[SearchHit]]:
        if not re.fullmatch(r"[a-f0-9]{32}", session_id) or not re.fullmatch(
            r"[a-f0-9]{64}", evidence_hash
        ):
            raise GroundingError("Invalid evidence session identity or digest.")
        record = await asyncio.to_thread(read_private_json, self.directory / f"{session_id}.json")
        if not isinstance(record, dict) or _digest(record) != evidence_hash:
            raise GroundingError("Evidence session integrity check failed.")
        if set(record) != {"version", "created_at", "binding", "question", "hits"}:
            raise GroundingError("Evidence session structure is invalid.")
        created = record["created_at"]
        if (
            record["version"] != 1
            or record["binding"] != self._binding()
            or not isinstance(created, (int, float))
            or not 0 <= time.time() - created <= SESSION_TTL_SECONDS
            or not isinstance(record["question"], str)
            or not isinstance(record["hits"], list)
            or not 1 <= len(record["hits"]) <= 50
        ):
            raise GroundingError("Evidence session is expired or its runtime binding changed.")
        try:
            hits = [SearchHit.model_validate(hit) for hit in record["hits"]]
        except ValidationError as exc:
            raise GroundingError("Evidence session contains invalid sources.") from exc
        if len({hit.chunk.id for hit in hits}) != len(hits):
            raise GroundingError("Evidence session has duplicate source identities.")
        current = await self.generator.store.get_hits_by_ids([hit.chunk.id for hit in hits])
        live = {hit.chunk.id: hit for hit in current}
        if len(live) != len(hits) or any(
            hit.chunk.id not in live
            or hit.chunk != live[hit.chunk.id].chunk
            or hit.source_uri != live[hit.chunk.id].source_uri
            for hit in hits
        ):
            raise GroundingError("Evidence sources were changed or deleted; retrieve again.")
        question = record["question"]
        if not question.strip() or len(self.generator._tokens(question)) > (
            self.generator.settings.max_question_tokens
        ):
            raise GroundingError("Evidence question exceeds its bounds.")
        if (
            self.generator._select_evidence(question, hits, self.generator._synthesis_schema())
            != hits
        ):
            raise GroundingError("Evidence no longer fits the bounded retrieval contract.")
        return question, hits

    def _proposal(self, raw: Any, hits: list[SearchHit]) -> tuple[AnswerProposal, str, list[str]]:
        try:
            proposal = AnswerProposal.model_validate(raw)
        except ValidationError as exc:
            raise GroundingError("Proposal does not match the answer schema.") from exc
        if not proposal.answerable:
            if proposal.claims or proposal.connections:
                raise GroundingError("An unanswerable proposal must have no answer items.")
            return proposal, "", []
        self.generator._validate_proposal(proposal, {hit.chunk.id: hit for hit in hits})
        markdown, ids = self.generator._render(proposal, {hit.chunk.id: hit for hit in hits})
        if len(self.generator._tokens(markdown)) > self.generator.settings.max_answer_tokens:
            raise GroundingError("Answer exceeds its token budget.")
        prose = " ".join(
            [claim.text for claim in proposal.claims]
            + [f"{item.concept_a} {item.concept_b} {item.text}" for item in proposal.connections]
        )
        if not self.generator._copying_within_limit(prose, hits):
            raise GroundingError("Answer exceeds the source reproduction budget.")
        return proposal, markdown, ids

    async def review_context(
        self, session_id: str, evidence_hash: str, proposal_path: Path
    ) -> dict[str, Any]:
        question, hits = await self._load(session_id, evidence_hash)
        raw = await asyncio.to_thread(read_private_json, proposal_path)
        proposal, _, _ = self._proposal(raw, hits)
        payload = {**self.generator._payload(question, hits), "proposal": proposal.model_dump()}
        schema = GroundingReview.model_json_schema()
        if self.generator._request_cost(SELF_REFLECTION_PROMPT, payload, schema) > (
            self.generator.settings.context_token_budget
        ):
            raise GroundingError("Review context exceeds its token budget.")
        return {
            "status": "awaiting_independent_review",
            "evidence_hash": evidence_hash,
            "proposal_hash": _digest(proposal.model_dump()),
            "verification_system_prompt": SELF_REFLECTION_PROMPT,
            "review_schema": schema,
            "payload": payload,
            "review_file_format": {
                "evidence_hash": evidence_hash,
                "proposal_hash": _digest(proposal.model_dump()),
                "review": "Object matching review_schema, produced by an independent reviewer.",
            },
        }

    async def verify(
        self, session_id: str, evidence_hash: str, proposal_path: Path, review_path: Path
    ) -> Answer:
        question, hits = await self._load(session_id, evidence_hash)
        raw = await asyncio.to_thread(read_private_json, proposal_path)
        proposal, markdown, ids = self._proposal(raw, hits)
        if not proposal.answerable:
            return self.generator._abstain("unanswerable")
        envelope = await asyncio.to_thread(read_private_json, review_path)
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"evidence_hash", "proposal_hash", "review"}
            or envelope["evidence_hash"] != evidence_hash
            or envelope["proposal_hash"] != _digest(proposal.model_dump())
        ):
            raise GroundingError("Independent review is not bound to this evidence and proposal.")
        try:
            review = GroundingReview.model_validate(envelope["review"])
        except ValidationError as exc:
            raise GroundingError("Independent review does not match its verdict schema.") from exc
        payload = {**self.generator._payload(question, hits), "proposal": proposal.model_dump()}
        if (
            self.generator._request_cost(
                SELF_REFLECTION_PROMPT, payload, GroundingReview.model_json_schema()
            )
            > self.generator.settings.context_token_budget
        ):
            raise GroundingError("Review context exceeds its token budget.")
        if (
            not review.approved
            or not self.generator._review_covers(
                review.claim_verdicts, {x.id for x in proposal.claims}
            )
            or not self.generator._review_covers(
                review.connection_verdicts, {x.id for x in proposal.connections}
            )
        ):
            return self.generator._abstain("review_rejected")
        return Answer(markdown=markdown, abstained=False, source_ids=ids)
