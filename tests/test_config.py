from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from mimir_rag.config import Settings
from mimir_rag.errors import ConfigurationError


def test_embedding_identity_and_chunking_version_are_distinct() -> None:
    original = Settings()
    chunked = Settings(chunk_target_tokens=500)
    assert original.embedding_fingerprint == chunked.embedding_fingerprint
    assert original.ingest_signature != chunked.ingest_signature
    assert (
        original.embedding_fingerprint != Settings(embedding_dimensions=512).embedding_fingerprint
    )


def test_invalid_chunk_limits_and_dimensions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_overlap_tokens=450)
    with pytest.raises(ValidationError):
        Settings(embedding_dimensions=2000)


def test_library_capacity_has_an_explicit_finite_upper_bound() -> None:
    assert Settings().max_chunks == 50_000
    assert Settings(max_chunks=150_000).max_chunks == 150_000
    for capacity in (0, 150_001):
        with pytest.raises(ValidationError):
            Settings(max_chunks=capacity)


def test_secrets_do_not_appear_in_serialization_or_repr() -> None:
    settings = Settings(openai_api_key=SecretStr("synthetic-private-value"))
    assert "synthetic-private-value" not in repr(settings)
    assert "openai_api_key" not in settings.model_dump()


def test_environment_provider_defaults_and_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMIR_SYNTHESIS_PROVIDER", "anthropic")
    monkeypatch.delenv("MIMIR_SYNTHESIS_MODEL", raising=False)
    assert Settings.from_env().synthesis_model == "claude-haiku-4-5-20251001"
    monkeypatch.setenv("MIMIR_DB_PATH", "~/mimir-test/library.sqlite3")
    assert Settings.from_env().db_path == Path.home() / "mimir-test" / "library.sqlite3"
    monkeypatch.setenv("MIMIR_EMBEDDING_DIMENSIONS", "invalid-secret-like-input")
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env()
    assert "embedding_dimensions" in str(error.value)
    assert "invalid-secret-like-input" not in str(error.value)
