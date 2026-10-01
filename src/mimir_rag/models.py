from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Document(Record):
    id: str
    source_uri: str
    content_hash: str
    generation: str
    title: str = Field(min_length=1, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    rights: str = Field(max_length=2000)
    metadata_origin: str
    ingest_signature: str


class Chunk(Record):
    id: str
    document_id: str
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    section: str = Field(max_length=1000)
    page: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    tokens: int = Field(gt=0)
    tags: list[str]
    concepts: list[str]

    @model_validator(mode="after")
    def validate_span(self) -> Chunk:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("Chunk character span must exactly match its text length.")
        if self.line_start is not None and self.line_end is not None:
            if self.line_end < self.line_start:
                raise ValueError("Chunk line span is reversed.")
        return self


class SearchHit(Record):
    chunk: Chunk
    score: float
    dense_score: float
    lexical_rank: int | None
    source_uri: str


class IngestResult(Record):
    document_id: str
    status: Literal["indexed", "unchanged"]
    chunk_count: int


class Answer(Record):
    markdown: str
    abstained: bool
    source_ids: list[str]
    reason: str | None = None
