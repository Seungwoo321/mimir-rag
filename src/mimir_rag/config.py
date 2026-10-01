from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from .errors import ConfigurationError


def _default_db_path() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "mimir-rag" / "library.sqlite3"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    db_path: Path = Field(default_factory=_default_db_path)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    anthropic_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1, le=3072)
    synthesis_provider: Literal["openai", "anthropic"] = "openai"
    synthesis_model: str = "gpt-4.1-mini"
    verification_model: str | None = None
    chunk_target_tokens: int = Field(default=450, ge=16, le=8000)
    chunk_max_tokens: int = Field(default=700, ge=16, le=8000)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    api_timeout_seconds: float = Field(default=45, gt=0, le=300)
    api_max_retries: int = Field(default=3, ge=0, le=5)
    api_concurrency: int = Field(default=4, ge=1, le=16)
    api_backoff_seconds: float = Field(default=0.5, ge=0, le=5)
    max_file_bytes: int = Field(default=25_000_000, ge=1)
    max_pdf_pages: int = Field(default=500, ge=1)
    max_pdf_stream_bytes: int = Field(default=8_000_000, ge=1)
    max_extracted_chars: int = Field(default=2_000_000, ge=1)
    max_chunks: int = Field(default=50_000, ge=1, le=50_000)
    sqlite_busy_timeout_ms: int = Field(default=5000, ge=1, le=60_000)
    dense_batch_size: int = Field(default=512, ge=1, le=4096)
    top_k: int = Field(default=6, ge=1, le=50)
    candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1)
    min_dense_score: float = Field(default=0.25, ge=-1, le=1)
    context_token_budget: int = Field(default=6000, ge=256, le=100_000)
    max_question_tokens: int = Field(default=1000, ge=1, le=8191)
    max_answer_claims: int = Field(default=8, ge=1, le=20)
    max_answer_tokens: int = Field(default=2048, ge=128, le=8192)
    max_verbatim_tokens: int = Field(default=48, ge=8, le=256)

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        if not 0 <= self.chunk_overlap_tokens < self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("Require overlap < target <= maximum chunk tokens.")
        if self.embedding_batch_size * self.chunk_max_tokens > 300_000:
            raise ValueError("Embedding batch token budget exceeds 300,000.")
        if self.embedding_model == "text-embedding-3-small" and self.embedding_dimensions > 1536:
            raise ValueError("text-embedding-3-small supports at most 1536 dimensions.")
        if not self.embedding_model.strip() or not self.synthesis_model.strip():
            raise ValueError("Model names cannot be empty.")
        object.__setattr__(self, "db_path", self.db_path.expanduser().absolute())
        return self

    @property
    def embedding_fingerprint(self) -> str:
        return json.dumps(
            {
                "provider": "openai",
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "encoding": "float32-le-l2-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def ingest_signature(self) -> str:
        payload = {
            "chunker": "structure-v1",
            "target": self.chunk_target_tokens,
            "maximum": self.chunk_max_tokens,
            "overlap": self.chunk_overlap_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_env(cls) -> Self:
        values: dict[str, object] = {}
        for field in cls.model_fields:
            key = field.upper() if field.endswith("api_key") else f"MIMIR_{field.upper()}"
            if key in os.environ:
                values[field] = os.environ[key]
        if values.get("synthesis_provider") == "anthropic" and "synthesis_model" not in values:
            values["synthesis_model"] = "claude-haiku-4-5-20251001"
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            fields = ", ".join(".".join(str(p) for p in item["loc"]) for item in exc.errors())
            raise ConfigurationError(
                f"Invalid settings: {fields}. Check MIMIR_ environment values."
            ) from None
