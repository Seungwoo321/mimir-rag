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
        Settings(embedding_dimensions=3073)


def test_library_capacity_has_an_explicit_finite_upper_bound() -> None:
    assert Settings().max_chunks == 50_000
    assert Settings(max_chunks=150_000).max_chunks == 150_000
    for capacity in (0, 150_001):
        with pytest.raises(ValidationError):
            Settings(max_chunks=capacity)


def test_secrets_do_not_appear_in_serialization_or_repr() -> None:
    settings = Settings(anthropic_api_key=SecretStr("synthetic-private-value"))
    assert "synthetic-private-value" not in repr(settings)
    assert "anthropic_api_key" not in settings.model_dump()


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


def test_local_default_is_pinned_and_openai_configuration_is_rejected() -> None:
    settings = Settings()
    assert settings.embedding_dimensions == 384
    assert settings.synthesis_provider == "claude-code"
    assert len(settings.embedding_revision) == 40
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert (
        settings.embedding_fingerprint
        != Settings(embedding_window_tokens=256).embedding_fingerprint
    )
    with pytest.raises(ValidationError):
        Settings.model_validate({"synthesis_provider": "openai"})
    with pytest.raises(ValidationError):
        Settings.model_validate({"embedding_model": "text-embedding-3-small"})
    with pytest.raises(ValidationError):
        Settings.model_validate({"openai_api_key": "secret"})


@pytest.mark.parametrize("field", ["synthesis_model", "verification_model"])
def test_non_claude_synthesis_model_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "gpt-4.1-mini"})
