"""Tests for settings loading, including the shipped .env template."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import Settings, secret_value

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_the_shipped_env_template_loads(tmp_path: Path, monkeypatch) -> None:
    """`cp .env.example .env` is step 2 of the README; it must not fail to boot.

    Blank optional values (`ANSWER_CACHE_TTL_S=`) reach pydantic as "", which is
    neither a float nor None, and used to raise a ValidationError at startup.
    """
    env = tmp_path / ".env"
    env.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in ("ANSWER_CACHE_TTL_S", "WHISPER_LANGUAGE", "GROQ_API_KEY", "EMBEDDING_MODEL"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.answer_cache_ttl_s is None
    assert settings.whisper_language is None
    assert settings.embedding_model is None
    assert settings.groq_api_key is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_optional_values_mean_unset(blank: str, monkeypatch) -> None:
    monkeypatch.setenv("ANSWER_CACHE_TTL_S", blank)
    monkeypatch.setenv("EMBEDDING_DIMENSION", blank)

    settings = Settings(_env_file=None)

    assert settings.answer_cache_ttl_s is None
    assert settings.embedding_dimension is None


def test_api_keys_are_not_printed_in_a_settings_dump(monkeypatch) -> None:
    """A stray repr() or a pydantic error must not leak a credential."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-super-secret-value")

    settings = Settings(_env_file=None)

    assert isinstance(settings.groq_api_key, SecretStr)
    assert "gsk-super-secret-value" not in repr(settings)
    assert "gsk-super-secret-value" not in str(settings.model_dump())
    assert secret_value(settings.groq_api_key) == "gsk-super-secret-value"


def test_secret_value_preserves_none() -> None:
    assert secret_value(None) is None
